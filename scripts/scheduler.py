#!/usr/bin/env python3
"""
Job scheduler for debate batch processing.

Handles:
- Concurrent debate processing
- Automatic retry logic
- Job status tracking
- Resource management
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class DebateJob:
    """Represents a single debate job."""

    job_id: str
    question: Dict[str, Any]
    agents: int = 5
    rounds: int = 2
    status: JobStatus = JobStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "question": self.question,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class DebateScheduler:
    """
    Manages debate job scheduling and execution.

    Features:
    - Concurrent execution with worker pool
    - Automatic retry on failures
    - Job status tracking
    - Progress reporting
    """

    def __init__(self, max_concurrent: int = 4, max_retries: int = 3, retry_delay: float = 5.0):
        """
        Initialize scheduler.

        Args:
            max_concurrent: Maximum concurrent debates
            max_retries: Maximum retry attempts per job
            retry_delay: Delay in seconds before retry
        """
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.jobs: Dict[str, DebateJob] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.active_jobs: int = 0
        self.completed_count: int = 0
        self.failed_count: int = 0

        logger.info(
            f"Scheduler initialized: max_concurrent={max_concurrent}, max_retries={max_retries}"
        )

    async def add_job(
        self,
        question: Dict[str, Any],
        job_id: Optional[str] = None,
        agents: int = 5,
        rounds: int = 2,
    ) -> str:
        """
        Add a debate job to the queue.

        Args:
            question: Question dictionary
            job_id: Optional custom job ID
            agents: Number of agents
            rounds: Number of rounds

        Returns:
            Job ID
        """
        if job_id is None:
            job_id = f"debate_{len(self.jobs):04d}"

        job = DebateJob(
            job_id=job_id,
            question=question,
            agents=agents,
            rounds=rounds,
            max_attempts=self.max_retries,
        )

        self.jobs[job_id] = job
        await self.queue.put(job)

        logger.info(f"Job {job_id} added to queue")
        return job_id

    async def add_batch(
        self, questions: List[Dict[str, Any]], agents: int = 5, rounds: int = 2
    ) -> List[str]:
        """
        Add multiple jobs at once.

        Args:
            questions: List of question dictionaries
            agents: Number of agents
            rounds: Number of rounds

        Returns:
            List of job IDs
        """
        job_ids = []
        for i, question in enumerate(questions):
            job_id = await self.add_job(
                question, job_id=f"batch_{i:04d}", agents=agents, rounds=rounds
            )
            job_ids.append(job_id)

        logger.info(f"Added {len(job_ids)} jobs to queue")
        return job_ids

    async def process_job(self, job: DebateJob, debate_fn: Callable) -> bool:
        """
        Process a single debate job.

        Args:
            job: The job to process
            debate_fn: Async function to run debate

        Returns:
            True if successful, False if failed
        """
        job.status = JobStatus.RUNNING
        job.attempts += 1
        job.started_at = datetime.now()

        logger.info(f"Processing job {job.job_id} (attempt {job.attempts}/{job.max_attempts})")

        try:
            # Run the debate
            result = await debate_fn(job.question, job.agents, job.rounds)

            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.now()
            self.completed_count += 1

            duration = (job.completed_at - job.started_at).total_seconds()
            logger.info(f"Job {job.job_id} completed successfully in {duration:.1f}s")
            return True

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {str(e)}")
            job.error = str(e)

            # Retry logic
            if job.attempts < job.max_attempts:
                job.status = JobStatus.RETRYING
                logger.info(f"Job {job.job_id} will retry after {self.retry_delay}s")

                # Add delay before retry
                await asyncio.sleep(self.retry_delay)

                # Re-queue
                await self.queue.put(job)
                return False
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                self.failed_count += 1
                logger.error(f"Job {job.job_id} failed permanently after {job.attempts} attempts")
                return False

    async def worker(self, worker_id: int, debate_fn: Callable):
        """
        Worker coroutine to process jobs from queue.

        Args:
            worker_id: Worker identifier
            debate_fn: Debate processing function
        """
        logger.info(f"Worker {worker_id} started")

        while True:
            # Get job from queue
            job = await self.queue.get()

            if job is None:  # Poison pill to stop worker
                self.queue.task_done()
                break

            self.active_jobs += 1

            try:
                await self.process_job(job, debate_fn)
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing job {job.job_id}: {e}")
            finally:
                self.active_jobs -= 1
                self.queue.task_done()

        logger.info(f"Worker {worker_id} stopped")

    async def run(self, debate_fn: Callable, progress_callback: Optional[Callable] = None):
        """
        Run the scheduler with worker pool.

        Args:
            debate_fn: Async function to run debates
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Starting scheduler with {self.max_concurrent} workers")

        # Create worker pool
        workers = [
            asyncio.create_task(self.worker(i, debate_fn)) for i in range(self.max_concurrent)
        ]

        # Optional progress reporting
        if progress_callback:

            async def report_progress():
                while self.active_jobs > 0 or not self.queue.empty():
                    await progress_callback(self.get_status())
                    await asyncio.sleep(5)  # Report every 5 seconds

            progress_task = asyncio.create_task(report_progress())
        else:
            progress_task = None

        # Wait for all jobs to complete
        await self.queue.join()

        # Stop workers
        for _ in range(self.max_concurrent):
            await self.queue.put(None)  # Poison pill

        # Wait for workers to finish
        await asyncio.gather(*workers)

        # Stop progress reporting
        if progress_task:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"Scheduler completed: {self.completed_count} succeeded, {self.failed_count} failed"
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status.

        Returns:
            Status dictionary
        """
        total = len(self.jobs)
        pending = sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING)
        running = sum(1 for j in self.jobs.values() if j.status == JobStatus.RUNNING)
        retrying = sum(1 for j in self.jobs.values() if j.status == JobStatus.RETRYING)

        return {
            "total_jobs": total,
            "queue_size": self.queue.qsize(),
            "active_jobs": self.active_jobs,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "pending": pending,
            "running": running,
            "retrying": retrying,
            "progress_percent": (self.completed_count / total * 100) if total > 0 else 0,
        }

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all completed job results.

        Returns:
            List of result dictionaries
        """
        return [
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "result": job.result,
                "error": job.error,
                "attempts": job.attempts,
                "duration": (job.completed_at - job.started_at).total_seconds()
                if job.completed_at and job.started_at
                else None,
            }
            for job in self.jobs.values()
            if job.status == JobStatus.COMPLETED
        ]

    def get_failed_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all failed jobs.

        Returns:
            List of failed job dictionaries
        """
        return [
            {
                "job_id": job.job_id,
                "question": job.question,
                "error": job.error,
                "attempts": job.attempts,
            }
            for job in self.jobs.values()
            if job.status == JobStatus.FAILED
        ]
