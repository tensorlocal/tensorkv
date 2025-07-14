package vamana

import (
	"unsafe"
)

func euclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func euclideanDistanceUnroll(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("两个切片的长度不相等")
	}

	var sum1, sum2, sum3, sum4 float64
	n := len(a)

	for i := 0; i < n-3; i += 4 {
		diff1 := a[i] - b[i]
		sum1 += diff1 * diff1

		diff2 := a[i+1] - b[i+1]
		sum2 += diff2 * diff2

		diff3 := a[i+2] - b[i+2]
		sum3 += diff3 * diff3

		diff4 := a[i+3] - b[i+3]
		sum4 += diff4 * diff4
	}

	sum := sum1 + sum2 + sum3 + sum4
	for i := n - (n % 4); i < n; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sum
}

func euclideanDistanceUnsafe(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}

	n := len(a)
	var sum float64

	// Get pointers to the underlying arrays
	aPtr := (*float64)(unsafe.Pointer(&a[0]))
	bPtr := (*float64)(unsafe.Pointer(&b[0]))

	// Process 4 elements at a time with pointer arithmetic
	i := 0
	for i <= n-4 {
		a0 := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i)*8))
		b0 := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr(i)*8))
		a1 := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i+1)*8))
		b1 := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr(i+1)*8))
		a2 := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i+2)*8))
		b2 := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr(i+2)*8))
		a3 := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i+3)*8))
		b3 := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr(i+3)*8))

		diff0 := a0 - b0
		diff1 := a1 - b1
		diff2 := a2 - b2
		diff3 := a3 - b3

		sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
		i += 4
	}

	// Handle remaining elements
	for i < n {
		ai := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i)*8))
		bi := *(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr(i)*8))
		diff := ai - bi
		sum += diff * diff
		i++
	}

	return sum
}

func euclideanDistanceUltraFast4(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}

	n := len(a)
	var sum float64

	// 大块处理，提高缓存利用率
	blockSize := 64 // 适合L1缓存

	for start := 0; start < n; start += blockSize {
		end := start + blockSize
		if end > n {
			end = n
		}

		// 块内8路展开
		i := start
		for i <= end-8 {
			diff0 := a[i] - b[i]
			diff1 := a[i+1] - b[i+1]
			diff2 := a[i+2] - b[i+2]
			diff3 := a[i+3] - b[i+3]
			diff4 := a[i+4] - b[i+4]
			diff5 := a[i+5] - b[i+5]
			diff6 := a[i+6] - b[i+6]
			diff7 := a[i+7] - b[i+7]

			sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3 +
				diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7
			i += 8
		}

		// 处理块内剩余元素
		for i < end {
			diff := a[i] - b[i]
			sum += diff * diff
			i++
		}
	}

	return sum
}

/*

// EuclideanDistanceNEON 在 arm64 架构上提供极致性能
func euclideanDistanceARM64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("两个切片的长度不相等")
	}
	n := len(a)
	if n == 0 {
		return 0.0
	}

	// NEON 指令一次处理 2 个 float64，所以我们先处理能被 2 整除的部分
	neonLen := n / 2 * 2
	var sum float64

	if neonLen > 0 {
		// 调用汇编函数，它只处理 neonLen 个元素
		sum = euclideanDistanceSIMD_ARM64(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), uintptr(neonLen))
	}

	// Go 代码处理剩余的 "尾巴" 元素 (如果 n 是奇数)
	if n%2 != 0 {
		diff := a[n-1] - b[n-1]
		sum += diff * diff
	}

	return sum
}

//go:noescape
func euclideanDistanceSIMD_ARM64(a, b unsafe.Pointer, len uintptr) float64
*/
