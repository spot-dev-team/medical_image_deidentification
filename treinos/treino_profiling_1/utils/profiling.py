# -*- coding: utf-8 -*-
"""
utils/profiling.py — Módulo de Monitorização de Treino DDP
============================================================
Mede quatro categorias críticas para diagnóstico de treino distribuído:
  1. VRAM        — pico de memória por fase (forward, backward, optimizer step)
  2. Comunicação — overhead all-reduce / SyncBatchNorm via NCCL
  3. Starvation  — tempo que a GPU espera por dados do DataLoader
  4. Timeline    — breakdown temporal de cada fase do loop de treino

Uso:
  from utils.profiling import TrainingProfiler
  profiler = TrainingProfiler(save_dir, global_rank, enabled=True)

  with profiler.profile_epoch(epoch):
      for batch in profiler.iter_dataloader(loader):
          with profiler.phase("forward"):   ...
          with profiler.phase("backward"):  ...
          with profiler.phase("optimizer"): ...

  profiler.save_summary()     # CSV + TXT legível
  profiler.export_chrome()    # JSON para chrome://tracing
"""

import os
import time
import json
import logging
import contextlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Iterator

import torch
import torch.distributed as dist


# ─────────────────────────────────────────────────────────────────────────────
# 1. ESTRUTURAS DE DADOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseStats:
    """Acumula métricas de uma fase ao longo de um epoch inteiro."""
    name:             str
    total_time_ms:    float = 0.0
    call_count:       int   = 0
    peak_vram_mb:     float = 0.0   # pico durante esta fase (MB)
    vram_delta_mb:    float = 0.0   # alocação líquida introduzida por esta fase (MB)

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.call_count, 1)

    def to_dict(self) -> dict:
        return {
            "phase":          self.name,
            "total_ms":       round(self.total_time_ms, 2),
            "avg_ms":         round(self.avg_time_ms,   2),
            "calls":          self.call_count,
            "peak_vram_mb":   round(self.peak_vram_mb,  1),
            "vram_delta_mb":  round(self.vram_delta_mb, 1),
        }


@dataclass
class EpochReport:
    epoch:           int
    phases:          dict = field(default_factory=dict)   # {name: PhaseStats}
    loader_wait_ms:  float = 0.0   # tempo total à espera de dados (starvation)
    total_epoch_ms:  float = 0.0
    nccl_ms:         float = 0.0   # estimativa de tempo gasto em all-reduces
    batches:         int   = 0

    @property
    def gpu_utilization_pct(self) -> float:
        """Fracção do epoch com compute real (exclui starvation e comunicação)."""
        compute = sum(p.total_time_ms for p in self.phases.values())
        return 100.0 * compute / max(self.total_epoch_ms, 1)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Epoch {self.epoch:03d} | Batches: {self.batches} | Total: {self.total_epoch_ms/1000:.1f}s",
            f"  GPU Utilization (compute): {self.gpu_utilization_pct:.1f}%",
            f"  DataLoader Starvation:     {self.loader_wait_ms:.1f} ms  "
            f"({100*self.loader_wait_ms/max(self.total_epoch_ms,1):.1f}%)",
            f"  NCCL / Comm Overhead:      {self.nccl_ms:.1f} ms  "
            f"({100*self.nccl_ms/max(self.total_epoch_ms,1):.1f}%)",
        ]
        for ps in self.phases.values():
            lines.append(
                f"  Phase [{ps.name:12s}]  avg {ps.avg_time_ms:7.1f} ms/batch | "
                f"peak VRAM {ps.peak_vram_mb:6.0f} MB | Δ {ps.vram_delta_mb:+6.0f} MB"
            )
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# 2. HOOK DDP — mede tempo de cada all-reduce (gradients + SyncBatchNorm)
# ─────────────────────────────────────────────────────────────────────────────

class _CommTimerHook:
    """
    Regista-se como comm_hook no DDP para cronometrar cada operação all-reduce.

    O SyncBatchNorm dispara all-reduces ADICIONAIS durante o forward pass,
    além dos all-reduces normais de gradientes no backward. Ambos aparecem aqui.

    Nota: só mede o overhead de comunicação dos gradientes (DDP hook).
    O overhead do SyncBatchNorm no forward NÃO passa por este hook —
    é capturado pelo torch.profiler (operação ncclAllReduce no CUDA trace).
    """

    def __init__(self):
        self.accumulated_ms: float = 0.0

    def hook(self, process_group, bucket):
        t0 = time.perf_counter()
        fut = dist.all_reduce(bucket.buffer(), group=process_group, async_op=True).get_future()

        def _done(fut):
            self.accumulated_ms += (time.perf_counter() - t0) * 1000.0
            result = fut.value()
            # Nas versões novas, o PyTorch devolve uma lista: [tensor(...)]
            # Temos de extrair o tensor lá de dentro!
            return result[0] if isinstance(result, list) else result

        return fut.then(_done)

    def reset(self):
        self.accumulated_ms = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASSE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class TrainingProfiler:
    """
    Instrumentação não-invasiva para treino DDP.

    Parâmetros
    ----------
    save_dir     : pasta onde os relatórios são escritos
    global_rank  : rank global do processo (os restantes contribuem mas não escrevem)
    enabled      : desactivar em produção sem alterar código
    profile_torch: activar o torch.profiler completo (caro — usar max 3 epochs)
    profile_epoch: qual epoch deve ser traçado com o torch.profiler
    """

    def __init__(
        self,
        save_dir: str,
        global_rank: int,
        enabled: bool = True,
        profile_torch: bool = True,
        profile_epoch: int = 2,          # epoch (0-indexed) com trace completo
    ):
        self.save_dir     = Path(save_dir)
        self.rank         = global_rank
        self.enabled      = enabled
        self.do_torch_prof = profile_torch
        self.target_epoch  = profile_epoch

        self._current_epoch: Optional[EpochReport]   = None
        self._current_phase: Optional[str]           = None
        self._phase_t0:      Optional[float]         = None
        self._epoch_t0:      Optional[float]         = None
        self._comm_hook:     Optional[_CommTimerHook] = None
        self._torch_profiler = None
        self._all_reports:   list[EpochReport]       = []

        # Sumário CSV (aberto em modo append)
        if self.rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = self.save_dir / "profiling_summary.csv"
            with open(self._csv_path, "w") as f:
                f.write("epoch,phase,avg_ms,total_ms,peak_vram_mb,vram_delta_mb,"
                        "loader_starvation_ms,nccl_ms,gpu_util_pct\n")

    # ── registo do DDP model ───────────────────────────────────────────────

    def register_ddp_model(self, ddp_model) -> None:
        """
        Chama após wrap DDP para instalar o hook de temporização de all-reduces.
        Os all-reduces de gradientes (backward) passam aqui.
        """
        if not self.enabled:
            return
        self._comm_hook = _CommTimerHook()
        ddp_model.register_comm_hook(
            state=dist.GroupMember.WORLD,
            hook=self._comm_hook.hook,
        )
        if self.rank == 0:
            logging.info("[Profiler] Comm hook registado no modelo DDP.")

    # ── context manager por epoch ──────────────────────────────────────────

    @contextlib.contextmanager
    def profile_epoch(self, epoch: int):
        """Envolve um epoch inteiro. Grava relatório no final."""
        if not self.enabled:
            yield
            return

        self._current_epoch = EpochReport(epoch=epoch)
        self._epoch_t0 = time.perf_counter()
        if self._comm_hook:
            self._comm_hook.reset()

        # Activar torch.profiler apenas no epoch alvo (overhead significativo)
        use_torch_prof = self.do_torch_prof and (epoch == self.target_epoch)
        if use_torch_prof:
            self._torch_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,          # rastreia alocações de memória
                with_stack=False,             # desactivar para menor overhead
                on_trace_ready=self._on_trace_ready,
                schedule=torch.profiler.schedule(
                    wait=2,     # batches iniciais ignorados (warm-up JIT)
                    warmup=2,   # batches de aquecimento do profiler
                    active=10,  # batches efectivamente traçados
                    repeat=1,
                ),
            )
            self._torch_profiler.__enter__()
            if self.rank == 0:
                logging.info(f"[Profiler] torch.profiler ACTIVO no epoch {epoch} "
                             f"(batches 4-13 capturados).")

        try:
            yield
        finally:
            # Fechar torch.profiler se estava activo
            if use_torch_prof and self._torch_profiler:
                self._torch_profiler.__exit__(None, None, None)
                self._torch_profiler = None

            elapsed = (time.perf_counter() - self._epoch_t0) * 1000.0
            self._current_epoch.total_epoch_ms = elapsed

            if self._comm_hook:
                self._current_epoch.nccl_ms = self._comm_hook.accumulated_ms

            self._all_reports.append(self._current_epoch)
            self._flush_epoch_report(self._current_epoch)

    def _on_trace_ready(self, prof):
        """Callback do torch.profiler — exporta trace e tabelas."""
        if self.rank != 0:
            return
        epoch = self._current_epoch.epoch if self._current_epoch else 0

        # 1. Chrome trace (abrir em chrome://tracing ou Perfetto UI)
        trace_path = self.save_dir / f"chrome_trace_epoch{epoch:03d}.json"
        prof.export_chrome_trace(str(trace_path))
        logging.info(f"[Profiler] Chrome trace → {trace_path}")

        # 2. Tabela de operações mais demoradas
        key_table = prof.key_averages(group_by_input_shape=False).table(
            sort_by="cuda_time_total", row_limit=30
        )
        table_path = self.save_dir / f"op_table_epoch{epoch:03d}.txt"
        table_path.write_text(key_table)
        logging.info(f"[Profiler] Op table → {table_path}")

        # 3. Filtrar especificamente operações NCCL (SyncBN + gradients)
        nccl_lines = [
            l for l in key_table.split("\n")
            if any(k in l.lower() for k in ["nccl", "allreduce", "all_reduce",
                                             "sync_batch", "syncbatch"])
        ]
        if nccl_lines:
            nccl_path = self.save_dir / f"nccl_ops_epoch{epoch:03d}.txt"
            nccl_path.write_text(
                "Operações NCCL / SyncBatchNorm detectadas:\n"
                + "\n".join(nccl_lines)
            )
            logging.info(f"[Profiler] NCCL ops → {nccl_path}")

        # 4. Relatório de memória CUDA
        mem_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20)
        mem_path = self.save_dir / f"memory_table_epoch{epoch:03d}.txt"
        mem_path.write_text(mem_table)
        logging.info(f"[Profiler] Memory table → {mem_path}")

    # ── iterador de DataLoader com medição de starvation ──────────────────

    def iter_dataloader(self, loader) -> Iterator:
        """
        Substitui 'for batch in loader' por 'for batch in profiler.iter_dataloader(loader)'.
        Mede o tempo entre o fim de uma iteração e o início da próxima — esse
        intervalo corresponde ao tempo que a GPU ficou à espera do CPU worker.
        """
        if not self.enabled or self._current_epoch is None:
            yield from loader
            return

        wait_total_ms = 0.0
        batch_count   = 0
        iter_t0       = time.perf_counter()   # marca antes do primeiro batch

        for batch in loader:
            # O tempo desde o final do batch anterior até agora é starvation
            stall_ms = (time.perf_counter() - iter_t0) * 1000.0
            wait_total_ms += stall_ms
            batch_count   += 1

            yield batch

            # Sincroniza para ter um marco temporal real pós-compute
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iter_t0 = time.perf_counter()

        self._current_epoch.loader_wait_ms += wait_total_ms
        self._current_epoch.batches        += batch_count

        # Notificar se o starvation for elevado (>15% do tempo total por batch)
        if batch_count > 0 and self.rank == 0:
            avg_stall = wait_total_ms / batch_count
            if avg_stall > 50:   # >50 ms por batch é preocupante
                logging.warning(
                    f"[Profiler] ⚠  Starvation elevado: "
                    f"{avg_stall:.0f} ms/batch — "
                    f"considera aumentar num_workers ou pin_memory."
                )

    # ── context manager por fase ───────────────────────────────────────────

    @contextlib.contextmanager
    def phase(self, name: str):
        """
        Cronometra uma fase e regista pico de VRAM.
        Uso: with profiler.phase("forward"): ...
        """
        if not self.enabled or self._current_epoch is None:
            yield
            return

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.synchronize(device)

        # Memória antes
        mem_before_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()

        # Anotação no torch.profiler se activo
        label = f"REMPE/{name}"
        with torch.profiler.record_function(label) if self._torch_profiler else contextlib.nullcontext():
            yield

        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Memória depois
        mem_after_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        peak_mb      = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        # Acumular nos stats da fase
        ep = self._current_epoch
        if name not in ep.phases:
            ep.phases[name] = PhaseStats(name=name)
        ps = ep.phases[name]
        ps.total_time_ms += elapsed_ms
        ps.call_count    += 1
        ps.peak_vram_mb   = max(ps.peak_vram_mb, peak_mb)
        ps.vram_delta_mb += (mem_after_mb - mem_before_mb)

        # Avançar o torch.profiler (step após cada fase "optimizer" — final de cada batch)
        if name == "optimizer" and self._torch_profiler:
            self._torch_profiler.step()

    # ── snapshot de memória instantâneo (usar a qualquer momento) ─────────

    def log_memory_checkpoint(self, label: str = "") -> dict:
        """
        Regista um snapshot de memória e devolve dict com as métricas.
        Útil para chamar antes/depois do modelo, antes do DataLoader, etc.
        """
        if not self.enabled:
            return {}
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        allocated = torch.cuda.memory_allocated(dev)  / 1024 ** 2
        reserved  = torch.cuda.memory_reserved(dev)   / 1024 ** 2
        free      = (torch.cuda.get_device_properties(dev).total_memory
                     - torch.cuda.memory_reserved(dev)) / 1024 ** 2
        stats = {
            "label":        label,
            "allocated_mb": round(allocated, 1),
            "reserved_mb":  round(reserved,  1),
            "free_mb":      round(free,      1),
        }
        if self.rank == 0:
            logging.info(
                f"[VRAM] {label:30s} | "
                f"Alocado: {allocated:6.0f} MB | "
                f"Reservado: {reserved:6.0f} MB | "
                f"Livre: {free:6.0f} MB"
            )
        return stats

    # ── flush do relatório de epoch ────────────────────────────────────────

    def _flush_epoch_report(self, report: EpochReport) -> None:
        if self.rank != 0:
            return

        # Log legível no terminal
        for line in report.summary_lines():
            logging.info(line)

        # Guardar em TXT acumulativo
        txt_path = self.save_dir / "profiling_log.txt"
        with open(txt_path, "a") as f:
            f.write("\n".join(report.summary_lines()) + "\n\n")

        # Guardar em CSV por fase
        with open(self._csv_path, "a") as f:
            for ps in report.phases.values():
                f.write(
                    f"{report.epoch},{ps.name},{ps.avg_time_ms:.2f},"
                    f"{ps.total_time_ms:.2f},{ps.peak_vram_mb:.1f},"
                    f"{ps.vram_delta_mb:.1f},"
                    f"{report.loader_wait_ms:.1f},"
                    f"{report.nccl_ms:.1f},"
                    f"{report.gpu_utilization_pct:.1f}\n"
                )

    # ── relatório final após treino ────────────────────────────────────────

    def save_final_report(self) -> None:
        """Gera análise agregada sobre todos os epochs. Chamar após o loop de treino."""
        if self.rank != 0 or not self.enabled or not self._all_reports:
            return

        path = self.save_dir / "profiling_final_report.txt"
        lines = [
            "=" * 70,
            "RELATÓRIO FINAL DE PROFILING — REMPE DDP",
            "=" * 70,
            "",
        ]

        # Métricas médias por fase
        phase_names = set(
            name for r in self._all_reports for name in r.phases.keys()
        )
        lines.append("MÉDIAS POR FASE (sobre todos os epochs):")
        for pname in sorted(phase_names):
            times  = [r.phases[pname].avg_time_ms for r in self._all_reports if pname in r.phases]
            vramps = [r.phases[pname].peak_vram_mb for r in self._all_reports if pname in r.phases]
            if times:
                lines.append(
                    f"  {pname:12s} → {sum(times)/len(times):.1f} ms/batch | "
                    f"Peak VRAM {max(vramps):.0f} MB"
                )

        # Starvation e NCCL médios
        avg_stv  = sum(r.loader_wait_ms for r in self._all_reports) / len(self._all_reports)
        avg_nccl = sum(r.nccl_ms        for r in self._all_reports) / len(self._all_reports)
        avg_util = sum(r.gpu_utilization_pct for r in self._all_reports) / len(self._all_reports)

        lines += [
            "",
            f"DataLoader Starvation (média):  {avg_stv:.0f} ms/epoch",
            f"NCCL Gradient Comm (média):     {avg_nccl:.0f} ms/epoch",
            f"GPU Utilização (compute):       {avg_util:.1f}%",
            "",
            "DIAGNÓSTICO SyncBatchNorm:",
            "  Para avaliar o overhead do SyncBatchNorm, abre o Chrome trace",
            "  (chrome://tracing) e filtra por 'ncclAllReduce'.",
            "  - Chamadas durante o FORWARD  → SyncBatchNorm (overhead adicional)",
            "  - Chamadas durante o BACKWARD → all-reduce de gradientes (normal DDP)",
            "  Se vires >50% das chamadas ncclAllReduce no forward, o excerto do",
            "  Gemini tem fundamento para a tua arquitectura.",
            "=" * 70,
        ]

        path.write_text("\n".join(lines))
        logging.info(f"[Profiler] Relatório final → {path}")
