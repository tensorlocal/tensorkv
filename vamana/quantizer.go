package vamana

type Quantizer interface {
	Train(vectors [][]float64)
	Compress(vector []float64) []byte
	ApproximateDistance(query []float64, compressed []byte) float64
	CompressedVectorSize() int
}
