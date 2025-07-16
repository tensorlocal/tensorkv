package vamana

import (
	"math"
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

func euclideanDistanceOptimized(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}

	n := len(a)
	var sum float64

	// Get direct pointers to slice data
	aPtr := uintptr(unsafe.Pointer(&a[0]))
	bPtr := uintptr(unsafe.Pointer(&b[0]))

	// Process 8 elements at a time for better cache utilization
	i := 0
	for i <= n-8 {
		// Load 8 pairs of values
		a0 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i)*8))
		b0 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i)*8))
		a1 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+1)*8))
		b1 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+1)*8))
		a2 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+2)*8))
		b2 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+2)*8))
		a3 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+3)*8))
		b3 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+3)*8))
		a4 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+4)*8))
		b4 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+4)*8))
		a5 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+5)*8))
		b5 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+5)*8))
		a6 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+6)*8))
		b6 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+6)*8))
		a7 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+7)*8))
		b7 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+7)*8))

		// Calculate differences
		diff0 := a0 - b0
		diff1 := a1 - b1
		diff2 := a2 - b2
		diff3 := a3 - b3
		diff4 := a4 - b4
		diff5 := a5 - b5
		diff6 := a6 - b6
		diff7 := a7 - b7

		// Accumulate squares (order optimized for instruction pipelining)
		sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3 +
			diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7
		i += 8
	}

	// Process remaining 4 elements
	for i <= n-4 {
		a0 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i)*8))
		b0 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i)*8))
		a1 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+1)*8))
		b1 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+1)*8))
		a2 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+2)*8))
		b2 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+2)*8))
		a3 := *(*float64)(unsafe.Pointer(aPtr + uintptr(i+3)*8))
		b3 := *(*float64)(unsafe.Pointer(bPtr + uintptr(i+3)*8))

		diff0 := a0 - b0
		diff1 := a1 - b1
		diff2 := a2 - b2
		diff3 := a3 - b3

		sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
		i += 4
	}

	// Handle remaining elements
	for i < n {
		ai := *(*float64)(unsafe.Pointer(aPtr + uintptr(i)*8))
		bi := *(*float64)(unsafe.Pointer(bPtr + uintptr(i)*8))
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

func euclideanDistanceUltraOptimized(a, b []float64) float64 {
	n := len(a)
	if n == 0 {
		return 0
	}

	// 使用更多累加器减少依赖链，并提前计算指针偏移
	var sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8 float64
	aPtr := uintptr(unsafe.Pointer(&a[0]))
	bPtr := uintptr(unsafe.Pointer(&b[0]))

	// 处理32个元素为一组，使用8个累加器
	i := 0
	for i <= n-32 {
		// 使用更细粒度的展开和更多累加器
		base := uintptr(i) * 8

		// 第一组4个元素
		a0 := *(*float64)(unsafe.Pointer(aPtr + base))
		b0 := *(*float64)(unsafe.Pointer(bPtr + base))
		a1 := *(*float64)(unsafe.Pointer(aPtr + base + 8))
		b1 := *(*float64)(unsafe.Pointer(bPtr + base + 8))
		a2 := *(*float64)(unsafe.Pointer(aPtr + base + 16))
		b2 := *(*float64)(unsafe.Pointer(bPtr + base + 16))
		a3 := *(*float64)(unsafe.Pointer(aPtr + base + 24))
		b3 := *(*float64)(unsafe.Pointer(bPtr + base + 24))

		diff0 := a0 - b0
		diff1 := a1 - b1
		diff2 := a2 - b2
		diff3 := a3 - b3

		sum1 += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3

		// 第二组4个元素
		a4 := *(*float64)(unsafe.Pointer(aPtr + base + 32))
		b4 := *(*float64)(unsafe.Pointer(bPtr + base + 32))
		a5 := *(*float64)(unsafe.Pointer(aPtr + base + 40))
		b5 := *(*float64)(unsafe.Pointer(bPtr + base + 40))
		a6 := *(*float64)(unsafe.Pointer(aPtr + base + 48))
		b6 := *(*float64)(unsafe.Pointer(bPtr + base + 48))
		a7 := *(*float64)(unsafe.Pointer(aPtr + base + 56))
		b7 := *(*float64)(unsafe.Pointer(bPtr + base + 56))

		diff4 := a4 - b4
		diff5 := a5 - b5
		diff6 := a6 - b6
		diff7 := a7 - b7

		sum2 += diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7

		// 继续第三到第八组...
		base += 64

		// 第三组
		a8 := *(*float64)(unsafe.Pointer(aPtr + base))
		b8 := *(*float64)(unsafe.Pointer(bPtr + base))
		a9 := *(*float64)(unsafe.Pointer(aPtr + base + 8))
		b9 := *(*float64)(unsafe.Pointer(bPtr + base + 8))
		a10 := *(*float64)(unsafe.Pointer(aPtr + base + 16))
		b10 := *(*float64)(unsafe.Pointer(bPtr + base + 16))
		a11 := *(*float64)(unsafe.Pointer(aPtr + base + 24))
		b11 := *(*float64)(unsafe.Pointer(bPtr + base + 24))

		diff8 := a8 - b8
		diff9 := a9 - b9
		diff10 := a10 - b10
		diff11 := a11 - b11

		sum3 += diff8*diff8 + diff9*diff9 + diff10*diff10 + diff11*diff11

		// 第四组
		a12 := *(*float64)(unsafe.Pointer(aPtr + base + 32))
		b12 := *(*float64)(unsafe.Pointer(bPtr + base + 32))
		a13 := *(*float64)(unsafe.Pointer(aPtr + base + 40))
		b13 := *(*float64)(unsafe.Pointer(bPtr + base + 40))
		a14 := *(*float64)(unsafe.Pointer(aPtr + base + 48))
		b14 := *(*float64)(unsafe.Pointer(bPtr + base + 48))
		a15 := *(*float64)(unsafe.Pointer(aPtr + base + 56))
		b15 := *(*float64)(unsafe.Pointer(bPtr + base + 56))

		diff12 := a12 - b12
		diff13 := a13 - b13
		diff14 := a14 - b14
		diff15 := a15 - b15

		sum4 += diff12*diff12 + diff13*diff13 + diff14*diff14 + diff15*diff15

		// 第五到第八组类似处理...
		base += 64

		// 第五组
		a16 := *(*float64)(unsafe.Pointer(aPtr + base))
		b16 := *(*float64)(unsafe.Pointer(bPtr + base))
		a17 := *(*float64)(unsafe.Pointer(aPtr + base + 8))
		b17 := *(*float64)(unsafe.Pointer(bPtr + base + 8))
		a18 := *(*float64)(unsafe.Pointer(aPtr + base + 16))
		b18 := *(*float64)(unsafe.Pointer(bPtr + base + 16))
		a19 := *(*float64)(unsafe.Pointer(aPtr + base + 24))
		b19 := *(*float64)(unsafe.Pointer(bPtr + base + 24))

		diff16 := a16 - b16
		diff17 := a17 - b17
		diff18 := a18 - b18
		diff19 := a19 - b19

		sum5 += diff16*diff16 + diff17*diff17 + diff18*diff18 + diff19*diff19

		// 剩余组...
		a20 := *(*float64)(unsafe.Pointer(aPtr + base + 32))
		b20 := *(*float64)(unsafe.Pointer(bPtr + base + 32))
		a21 := *(*float64)(unsafe.Pointer(aPtr + base + 40))
		b21 := *(*float64)(unsafe.Pointer(bPtr + base + 40))
		a22 := *(*float64)(unsafe.Pointer(aPtr + base + 48))
		b22 := *(*float64)(unsafe.Pointer(bPtr + base + 48))
		a23 := *(*float64)(unsafe.Pointer(aPtr + base + 56))
		b23 := *(*float64)(unsafe.Pointer(bPtr + base + 56))

		diff20 := a20 - b20
		diff21 := a21 - b21
		diff22 := a22 - b22
		diff23 := a23 - b23

		sum6 += diff20*diff20 + diff21*diff21 + diff22*diff22 + diff23*diff23

		base += 64

		a24 := *(*float64)(unsafe.Pointer(aPtr + base))
		b24 := *(*float64)(unsafe.Pointer(bPtr + base))
		a25 := *(*float64)(unsafe.Pointer(aPtr + base + 8))
		b25 := *(*float64)(unsafe.Pointer(bPtr + base + 8))
		a26 := *(*float64)(unsafe.Pointer(aPtr + base + 16))
		b26 := *(*float64)(unsafe.Pointer(bPtr + base + 16))
		a27 := *(*float64)(unsafe.Pointer(aPtr + base + 24))
		b27 := *(*float64)(unsafe.Pointer(bPtr + base + 24))

		diff24 := a24 - b24
		diff25 := a25 - b25
		diff26 := a26 - b26
		diff27 := a27 - b27

		sum7 += diff24*diff24 + diff25*diff25 + diff26*diff26 + diff27*diff27

		a28 := *(*float64)(unsafe.Pointer(aPtr + base + 32))
		b28 := *(*float64)(unsafe.Pointer(bPtr + base + 32))
		a29 := *(*float64)(unsafe.Pointer(aPtr + base + 40))
		b29 := *(*float64)(unsafe.Pointer(bPtr + base + 40))
		a30 := *(*float64)(unsafe.Pointer(aPtr + base + 48))
		b30 := *(*float64)(unsafe.Pointer(bPtr + base + 48))
		a31 := *(*float64)(unsafe.Pointer(aPtr + base + 56))
		b31 := *(*float64)(unsafe.Pointer(bPtr + base + 56))

		diff28 := a28 - b28
		diff29 := a29 - b29
		diff30 := a30 - b30
		diff31 := a31 - b31

		sum8 += diff28*diff28 + diff29*diff29 + diff30*diff30 + diff31*diff31

		i += 32
	}

	// 处理16个元素为一组
	for i <= n-16 {
		base := uintptr(i) * 8

		// 第一组4个元素
		a0 := *(*float64)(unsafe.Pointer(aPtr + base))
		b0 := *(*float64)(unsafe.Pointer(bPtr + base))
		a1 := *(*float64)(unsafe.Pointer(aPtr + base + 8))
		b1 := *(*float64)(unsafe.Pointer(bPtr + base + 8))
		a2 := *(*float64)(unsafe.Pointer(aPtr + base + 16))
		b2 := *(*float64)(unsafe.Pointer(bPtr + base + 16))
		a3 := *(*float64)(unsafe.Pointer(aPtr + base + 24))
		b3 := *(*float64)(unsafe.Pointer(bPtr + base + 24))

		diff0 := a0 - b0
		diff1 := a1 - b1
		diff2 := a2 - b2
		diff3 := a3 - b3

		sum1 += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3

		// 第二组4个元素
		a4 := *(*float64)(unsafe.Pointer(aPtr + base + 32))
		b4 := *(*float64)(unsafe.Pointer(bPtr + base + 32))
		a5 := *(*float64)(unsafe.Pointer(aPtr + base + 40))
		b5 := *(*float64)(unsafe.Pointer(bPtr + base + 40))
		a6 := *(*float64)(unsafe.Pointer(aPtr + base + 48))
		b6 := *(*float64)(unsafe.Pointer(bPtr + base + 48))
		a7 := *(*float64)(unsafe.Pointer(aPtr + base + 56))
		b7 := *(*float64)(unsafe.Pointer(bPtr + base + 56))

		diff4 := a4 - b4
		diff5 := a5 - b5
		diff6 := a6 - b6
		diff7 := a7 - b7

		sum2 += diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7

		// 第三组4个元素
		a8 := *(*float64)(unsafe.Pointer(aPtr + base + 64))
		b8 := *(*float64)(unsafe.Pointer(bPtr + base + 64))
		a9 := *(*float64)(unsafe.Pointer(aPtr + base + 72))
		b9 := *(*float64)(unsafe.Pointer(bPtr + base + 72))
		a10 := *(*float64)(unsafe.Pointer(aPtr + base + 80))
		b10 := *(*float64)(unsafe.Pointer(bPtr + base + 80))
		a11 := *(*float64)(unsafe.Pointer(aPtr + base + 88))
		b11 := *(*float64)(unsafe.Pointer(bPtr + base + 88))

		diff8 := a8 - b8
		diff9 := a9 - b9
		diff10 := a10 - b10
		diff11 := a11 - b11

		sum3 += diff8*diff8 + diff9*diff9 + diff10*diff10 + diff11*diff11

		// 第四组4个元素
		a12 := *(*float64)(unsafe.Pointer(aPtr + base + 96))
		b12 := *(*float64)(unsafe.Pointer(bPtr + base + 96))
		a13 := *(*float64)(unsafe.Pointer(aPtr + base + 104))
		b13 := *(*float64)(unsafe.Pointer(bPtr + base + 104))
		a14 := *(*float64)(unsafe.Pointer(aPtr + base + 112))
		b14 := *(*float64)(unsafe.Pointer(bPtr + base + 112))
		a15 := *(*float64)(unsafe.Pointer(aPtr + base + 120))
		b15 := *(*float64)(unsafe.Pointer(bPtr + base + 120))

		diff12 := a12 - b12
		diff13 := a13 - b13
		diff14 := a14 - b14
		diff15 := a15 - b15

		sum4 += diff12*diff12 + diff13*diff13 + diff14*diff14 + diff15*diff15

		i += 16
	}

	// 处理8个元素为一组
	for i <= n-8 {
		base := uintptr(i) * 8

		// 第一组4个元素
		a0 := *(*float64)(unsafe.Pointer(aPtr + base))
		b0 := *(*float64)(unsafe.Pointer(bPtr + base))
		a1 := *(*float64)(unsafe.Pointer(aPtr + base + 8))
		b1 := *(*float64)(unsafe.Pointer(bPtr + base + 8))
		a2 := *(*float64)(unsafe.Pointer(aPtr + base + 16))
		b2 := *(*float64)(unsafe.Pointer(bPtr + base + 16))
		a3 := *(*float64)(unsafe.Pointer(aPtr + base + 24))
		b3 := *(*float64)(unsafe.Pointer(bPtr + base + 24))

		diff0 := a0 - b0
		diff1 := a1 - b1
		diff2 := a2 - b2
		diff3 := a3 - b3

		sum1 += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3

		// 第二组4个元素
		a4 := *(*float64)(unsafe.Pointer(aPtr + base + 32))
		b4 := *(*float64)(unsafe.Pointer(bPtr + base + 32))
		a5 := *(*float64)(unsafe.Pointer(aPtr + base + 40))
		b5 := *(*float64)(unsafe.Pointer(bPtr + base + 40))
		a6 := *(*float64)(unsafe.Pointer(aPtr + base + 48))
		b6 := *(*float64)(unsafe.Pointer(bPtr + base + 48))
		a7 := *(*float64)(unsafe.Pointer(aPtr + base + 56))
		b7 := *(*float64)(unsafe.Pointer(bPtr + base + 56))

		diff4 := a4 - b4
		diff5 := a5 - b5
		diff6 := a6 - b6
		diff7 := a7 - b7

		sum2 += diff4*diff4 + diff5*diff5 + diff6*diff6 + diff7*diff7

		i += 8
	}

	// 处理4个元素为一组
	for i <= n-4 {
		base := uintptr(i) * 8

		a0 := *(*float64)(unsafe.Pointer(aPtr + base))
		b0 := *(*float64)(unsafe.Pointer(bPtr + base))
		a1 := *(*float64)(unsafe.Pointer(aPtr + base + 8))
		b1 := *(*float64)(unsafe.Pointer(bPtr + base + 8))
		a2 := *(*float64)(unsafe.Pointer(aPtr + base + 16))
		b2 := *(*float64)(unsafe.Pointer(bPtr + base + 16))
		a3 := *(*float64)(unsafe.Pointer(aPtr + base + 24))
		b3 := *(*float64)(unsafe.Pointer(bPtr + base + 24))

		diff0 := a0 - b0
		diff1 := a1 - b1
		diff2 := a2 - b2
		diff3 := a3 - b3

		sum1 += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
		i += 4
	}

	// 处理剩余元素
	for i < n {
		base := uintptr(i) * 8
		ai := *(*float64)(unsafe.Pointer(aPtr + base))
		bi := *(*float64)(unsafe.Pointer(bPtr + base))
		diff := ai - bi
		sum1 += diff * diff
		i++
	}

	totalSum := sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8
	return math.Sqrt(totalSum)
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
