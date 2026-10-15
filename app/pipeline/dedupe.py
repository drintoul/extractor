"""Near-duplicate detection via MinHash + LSH."""
from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
LSH_THRESHOLD = 0.85
SHINGLE_SIZE = 5


def _shingles(text: str) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i:i + SHINGLE_SIZE])
            for i in range(max(1, len(words) - SHINGLE_SIZE + 1))}


def find_duplicates(doc_ids: list[str], texts: list[str]) -> set[str]:
    """Returns doc_ids that are near-duplicates of an earlier doc (sorted input order wins)."""
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    dupes: set[str] = set()
    for doc_id, text in zip(doc_ids, texts):
        mh = MinHash(num_perm=NUM_PERM)
        for shingle in _shingles(text):
            mh.update(shingle.encode())
        if lsh.query(mh):
            dupes.add(doc_id)
        else:
            lsh.insert(doc_id, mh)
    return dupes
