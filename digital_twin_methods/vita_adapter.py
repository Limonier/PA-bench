"""Optional VitaBench custom memory classes; existing runner stays untouched."""
import os
import uuid
from functools import lru_cache

from vita.memory.base import BaseMemory
from .bench_inputs import vita_history
from .methods import HERE, AlignGenerator, create


@lru_cache(maxsize=1)
def _trained_generator(checkpoint, quantize):
    return AlignGenerator(checkpoint, quantize)


class DigitalTwinMemory(BaseMemory):
    method = None

    def __init__(self, language=None, **kwargs):
        super().__init__(language=language, **kwargs)
        self.backend = None

    def _construct(self):
        options = {}
        if self.method == 'gpg':
            options['task'] = 'vita'
        elif self.method == 'alignxplore_plus':
            checkpoint = os.environ.get('ALIGNXPLORE_CHECKPOINT')
            if not checkpoint:
                raise RuntimeError('Set ALIGNXPLORE_CHECKPOINT to the official model directory')
            options['generator'] = _trained_generator(checkpoint, os.getenv('ALIGNXPLORE_4BIT')=='1')
        elif self.method in ('mem0','omem'):
            options['storage'] = HERE/'runtime'/self.method/uuid.uuid4().hex
        return create(self.method, **options)

    def update(self, new_interactions, llm=None, llm_args=None, **kwargs):
        if new_interactions:
            if self.backend is None:
                self.backend = self._construct()
            self.backend.update(vita_history(new_interactions))
        return self.read()

    def read(self, query=None):
        return self.backend.retrieve(query or 'Current user preferences and relevant history') if self.backend is not None else ''

    def reset(self):
        if self.backend is not None and hasattr(self.backend,'close'):
            self.backend.close()
        self.backend = None


class GPGMemory(DigitalTwinMemory):
    method = 'gpg'


class AlignXploreMemory(DigitalTwinMemory):
    method = 'alignxplore_plus'


class PGMemMemory(DigitalTwinMemory):
    method = 'pgmem'


class OMemMemory(DigitalTwinMemory):
    method = 'omem'


class Mem0Memory(DigitalTwinMemory):
    method = 'mem0'
