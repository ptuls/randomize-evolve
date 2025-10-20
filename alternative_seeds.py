"""Alternative initial programs to seed diverse evolution.

Use these by manually adding them to the database at the start of evolution,
or by running separate evolution runs and combining results.
"""

# Seed 1: Cuckoo-inspired filter with displacement
CUCKOO_SEED = """
import hashlib

class Candidate:
    def __init__(self, key_bits, capacity, bits_per_item=10):
        self.key_bits = key_bits
        self.capacity = capacity
        self.buckets = [{} for _ in range(4)]  # 4 hash tables

    def add(self, item):
        h = hashlib.sha256(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
        # Try 4 different positions
        for i in range(4):
            pos = int.from_bytes(h[i:i+2], 'little') % self.capacity
            self.buckets[i][pos] = int.from_bytes(h[4:6], 'little')

    def query(self, item):
        h = hashlib.sha256(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
        fp = int.from_bytes(h[4:6], 'little')
        return any(
            self.buckets[i].get(int.from_bytes(h[i:i+2], 'little') % self.capacity) == fp
            for i in range(4)
        )

def candidate_factory(key_bits, capacity):
    return Candidate(key_bits, capacity)
"""

# Seed 2: Counting filter with quotients
QUOTIENT_SEED = """
import hashlib

class Candidate:
    def __init__(self, key_bits, capacity, bits_per_item=10):
        self.key_bits = key_bits
        self.capacity = capacity
        self.table = bytearray(capacity * 2)  # 2 bytes per slot

    def add(self, item):
        h = hashlib.md5(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
        quotient = int.from_bytes(h[:2], 'little') % self.capacity
        remainder = h[2]
        idx = quotient * 2
        self.table[idx:idx+2] = remainder.to_bytes(2, 'little')

    def query(self, item):
        h = hashlib.md5(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
        quotient = int.from_bytes(h[:2], 'little') % self.capacity
        remainder = h[2]
        idx = quotient * 2
        stored = int.from_bytes(self.table[idx:idx+2], 'little')
        return stored == int.from_bytes(remainder.to_bytes(2, 'little'), 'little')

def candidate_factory(key_bits, capacity):
    return Candidate(key_bits, capacity)
"""

# Seed 3: XOR-based filter
XOR_SEED = """
import hashlib

class Candidate:
    def __init__(self, key_bits, capacity, bits_per_item=10):
        self.key_bits = key_bits
        self.capacity = capacity
        self.xor_table = [0] * capacity

    def add(self, item):
        h = hashlib.blake2b(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
        for i in range(3):
            idx = int.from_bytes(h[i*2:i*2+2], 'little') % self.capacity
            self.xor_table[idx] ^= int.from_bytes(h[6:8], 'little')

    def query(self, item):
        h = hashlib.blake2b(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
        fp = int.from_bytes(h[6:8], 'little')
        xor_result = 0
        for i in range(3):
            idx = int.from_bytes(h[i*2:i*2+2], 'little') % self.capacity
            xor_result ^= self.xor_table[idx]
        return xor_result == fp

def candidate_factory(key_bits, capacity):
    return Candidate(key_bits, capacity)
"""
