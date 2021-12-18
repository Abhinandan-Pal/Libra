
def chunk2json(chunk):
    chunks = chunk.split('; ')
    ranges = dict()
    for chunkrange in chunks:
        chunked = chunkrange.split(' ∈ ')
        bounds = chunked[1].strip().replace('[', '').replace(']', '').split(', ')
        ranges[chunked[0]] = {
            'lower bound': float(bounds[0]),
            'upper bound': float(bounds[1])
        }
    partition = {
        'ranges': ranges
    }
    return partition
