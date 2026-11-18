"""bandextdb: the bandext edge bands (workstream 1's winning
conversion) combined with dbmean aggregation (the independently
Supported robust aggregator), for the interaction test on the
bandext base. Same machinery as convert_variants2.run: band 0 from
0 Hz, band 24 to Nyquist, aggregation = mean of per-segment dB.
"""
import convert_variants as cv
import convert_variants2 as cv2

if __name__ == "__main__":
    edges = list(cv.EDGES)
    edges[0] = (0.0, edges[0][1])
    edges[24] = (edges[24][0], 16000.0)
    cv2.run("bandextdb", ("hann", "constant", 2, 2, ("dbmean",)),
            edges=edges)
    print("ALL DONE", flush=True)
