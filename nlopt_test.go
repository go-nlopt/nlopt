package nlopt

import (
	"errors"
	"fmt"
	"math"
	"testing"
	"unsafe"
)

func TestAlgorithmName(t *testing.T) {
	name := AlgorithmName(LD_LBFGS_NOCEDAL)
	if got, want := name, "original L-BFGS code by Nocedal et al. (NOT COMPILED)"; got != want {
		t.Errorf("Expected algorithm='%s', got='%s'", want, got)
	}
}

func TestSrand(t *testing.T) {
	// TODO test for success
	Srand(uint64(200))
}

func TestSrandTime(t *testing.T) {
	// TODO test for success
	SrandTime()
}

func TestVersion(t *testing.T) {
	v := Version()
	if len(v) == 3 || v == "0.0.0" {
		t.Errorf("Expected version number got='%s'", v)
	}
}

func TestNewNLoptUNKNOWN(t *testing.T) {
	_, err := NewNLopt(NUM_ALGORITHMS, 1)
	if err == nil {
		t.Fatal("Expected error, got <nil>")
	}
}

func TestNewNLoptNoDim(t *testing.T) {
	o, err := NewNLopt(GN_DIRECT, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetDimension(), o.dim; got != want {
		t.Errorf("Expected dim='%d', got='%d'", want, got)
	}
}

func TestNLopt_Copy(t *testing.T) {
	o, err := NewNLopt(GN_DIRECT, 0)
	if err != nil {
		t.Fatal(err)
	}
	o1 := o.Copy()
	if got, want := uintptr(unsafe.Pointer(o.cOpt)), uintptr(unsafe.Pointer(o1.cOpt)); got == want {
		t.Errorf("Expected ptr='%d', got='%d'", want, got)
	}
	if got, want := o.GetDimension(), o1.GetDimension(); got != want {
		t.Errorf("Expected dim='%d', got='%d'", want, got)
	}
	if got, want := o.GetDimension(), o1.GetDimension(); got != want {
		t.Errorf("Expected dim='%d', got='%d'", want, got)
	}
	if got, want := o.GetAlgorithm(), o1.GetAlgorithm(); got != want {
		t.Errorf("Expected algorithm='%d', got='%d'", want, got)
	}
}

func TestNLopt_OptimizeMMA(t *testing.T) {
	opt, err := NewNLopt(LD_MMA, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Destroy()

	opt.SetLowerBounds([]float64{math.Inf(-1), 0.})

	var count int
	myfunc := func(x, gradient []float64) float64 {
		count++
		if len(gradient) > 0 {
			gradient[0] = 0.0
			gradient[1] = 0.5 / math.Sqrt(x[1])
		}
		return math.Sqrt(x[1])
	}

	myconstraint := func(x, gradient []float64, a, b float64) float64 {
		if len(gradient) > 0 {
			gradient[0] = 3 * a * math.Pow(a*x[0]+b, 2.)
			gradient[1] = -1.0
		}
		return math.Pow(a*x[0]+b, 3) - x[1]
	}

	opt.SetMinObjective(myfunc)
	opt.AddInequalityConstraint(func(x, gradient []float64) float64 { return myconstraint(x, gradient, 2., 0.) }, 1e-8)
	opt.AddInequalityConstraint(func(x, gradient []float64) float64 { return myconstraint(x, gradient, -1., 1.) }, 1e-8)
	opt.SetXtolRel(1e-4)

	x := []float64{1.234, 5.678}
	xopt, minf, err := opt.Optimize(x)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := opt.LastStatus(), "XTOL_REACHED"; got != want {
		t.Errorf("Expected last status='%s', got='%s'", want, got)
	}
	if got, want := count, 11; got != want {
		t.Errorf("Expected evaluations count='%d', got='%d'", want, got)
	}
	fmt.Printf("MMA: found minimum at f(%g,%g) = %0.10g\n", xopt[0], xopt[1], minf)
}

func TestNLopt_OptimizeCOBYLA(t *testing.T) {
	opt, err := NewNLopt(LN_COBYLA, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Destroy()

	opt.SetLowerBounds([]float64{math.Inf(-1), 0.})

	var count int
	myfunc := func(x, gradient []float64) float64 {
		count++
		if len(gradient) > 0 {
			gradient[0] = 0.0
			gradient[1] = 0.5 / math.Sqrt(x[1])
		}
		return math.Sqrt(x[1])
	}
	myconstraint := func(x, gradient []float64, a, b float64) float64 {
		if len(gradient) > 0 {
			gradient[0] = 3 * a * math.Pow(a*x[0]+b, 2.)
			gradient[1] = -1.0
		}
		return math.Pow(a*x[0]+b, 3) - x[1]
	}

	opt.SetMinObjective(myfunc)
	opt.AddInequalityConstraint(func(x, gradient []float64) float64 { return myconstraint(x, gradient, 2., 0.) }, 1e-8)
	opt.AddInequalityConstraint(func(x, gradient []float64) float64 { return myconstraint(x, gradient, -1., 1.) }, 1e-8)
	opt.SetXtolRel(0.)
	opt.SetStopVal(math.Sqrt(8./27.) + 1e-3)

	x := []float64{1.234, 5.678}
	xopt, minf, err := opt.Optimize(x)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := opt.LastStatus(), "STOPVAL_REACHED"; got != want {
		t.Errorf("Expected last status='%s', got='%s'", want, got)
	}
	if got, want := count, 25; got != want {
		t.Errorf("Expected evaluations count='%d', got='%d'", want, got)
	}
	fmt.Printf("COBYLA: found minimum at f(%g,%g) = %0.10g\n", xopt[0], xopt[1], minf)
}

func TestNLopt_OptimizeCOBYLA_MConstraint(t *testing.T) {
	opt, err := NewNLopt(LN_COBYLA, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer opt.Destroy()

	opt.SetLowerBounds([]float64{math.Inf(-1), 0.})

	var count int
	myfunc := func(x, gradient []float64) float64 {
		count++
		if len(gradient) > 0 {
			gradient[0] = 0.0
			gradient[1] = 0.5 / math.Sqrt(x[1])
		}
		return math.Sqrt(x[1])
	}
	myconstraintm := func(results, x, gradient, a, b []float64) {
		n := len(x)
		for i := 0; i < len(results); i++ {
			va := a[i/2]
			vb := b[i/2]
			if len(gradient) > 0 {
				gradient[i*n] = 3 * va * math.Pow(va*x[0]+vb, 2.)
				gradient[i*n+1] = -1.0
			}
			results[i] = math.Pow(va*x[0]+vb, 3) - x[1]
		}
	}

	opt.SetMinObjective(myfunc)
	opt.AddInequalityMConstraint(func(result, x, gradient []float64) {
		myconstraintm(result, x, gradient, []float64{2., 0.}, []float64{-1., 1.})
	}, []float64{1e-8, 1e-8})
	opt.SetXtolRel(0.)
	opt.SetStopVal(math.Sqrt(8./27.) + 1e-3)
	opt.SetLowerBounds([]float64{-10, 1e-6})
	opt.SetUpperBounds1(10.)
	opt.SetMaxEval(10000)
	opt.SetInitialStep([]float64{1.0, 10.0})

	x := []float64{1.234, 5.678}
	xopt, minf, err := opt.Optimize(x)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := opt.LastStatus(), "STOPVAL_REACHED"; got != want {
		t.Errorf("Expected last status='%s', got='%s'", want, got)
	}
	if got, want := count, 4; got != want {
		t.Errorf("Expected evaluations count='%d', got='%d'", want, got)
	}
	fmt.Printf("COBYLA-M: found minimum at f(%g,%g) = %0.10g\n", xopt[0], xopt[1], minf)
}

func TestNLopt_SetMinObjective(t *testing.T) {
	obj := func(x, gradient []float64) float64 {
		return math.Inf(-1)
	}
	o, err := NewNLopt(GN_DIRECT_L_RAND, 1)
	if err != nil {
		t.Fatal(err)
	}
	err = o.SetMinObjective(obj)
	if err != nil {
		t.Fatal(err)
	}
	if len(o.funcs) == 0 {
		t.Fatal(errors.New("Expected allocated func"))
	}
	if len(funcMap) == 0 {
		t.Fatal(errors.New("Expected allocated func"))
	}
	o.Destroy()
	if len(funcMap) != 0 {
		t.Fatal(errors.New("Expected no allocated funcs"))
	}
}

func TestNLopt_SetMaxObjective(t *testing.T) {
	obj := func(x, gradient []float64) float64 {
		return math.Inf(-1)
	}
	o, err := NewNLopt(GN_DIRECT_L_RAND, 1)
	if err != nil {
		t.Fatal(err)
	}
	err = o.SetMaxObjective(obj)
	if err != nil {
		t.Fatal(err)
	}
	if len(o.funcs) == 0 {
		t.Fatal(errors.New("Expected allocated func"))
	}
	if len(funcMap) == 0 {
		t.Fatal(errors.New("Expected allocated func"))
	}
	o.Destroy()
	if len(funcMap) != 0 {
		t.Fatal(errors.New("Expected no allocated funcs"))
	}
}

func TestNLopt_GetAlgorithm(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	if got, want := o.GetAlgorithm(), LD_AUGLAG; got != want {
		t.Errorf("Expected algorithm='%d', got='%d'", want, got)
	}
}

func TestNLopt_GetAlgorithmName(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	if got, want := o.GetAlgorithmName(), "Augmented Lagrangian method (local, derivative)"; got != want {
		t.Errorf("Expected algorithm name='%s', got='%s'", want, got)
	}
}

func TestNLopt_GetDimension(t *testing.T) {
	o, err := NewNLopt(GN_DIRECT_L_RAND, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	if got, want := o.GetDimension(), uint(10); got != want {
		t.Errorf("Expected dimension='%d', got='%d'", want, got)
	}
}

// constraints:

func TestNLopt_SetLowerBounds(t *testing.T) {
	o, err := NewNLopt(LD_MMA, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	lb := []float64{math.Inf(-1), 0.}
	err = o.SetLowerBounds(lb)
	if err != nil {
		t.Fatal(err)
	}
	glb, err := o.GetLowerBounds()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := glb[0], lb[0]; got != want {
		t.Errorf("Expected glb='%f', got='%f'", want, got)
	}
	if got, want := glb[1], lb[1]; got != want {
		t.Errorf("Expected glb='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetLowerBounds1(t *testing.T) {
	o, err := NewNLopt(LD_MMA, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	lb := -1.
	err = o.SetLowerBounds1(lb)
	if err != nil {
		t.Fatal(err)
	}
	glb, err := o.GetLowerBounds()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := glb[0], lb; got != want {
		t.Errorf("Expected lb='%f', got='%f'", want, got)
	}
	if got, want := glb[1], lb; got != want {
		t.Errorf("Expected lb='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetUpperBounds(t *testing.T) {
	o, err := NewNLopt(LD_MMA, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	ub := []float64{1., math.Inf(1.)}
	err = o.SetUpperBounds(ub)
	if err != nil {
		t.Fatal(err)
	}
	gub, err := o.GetUpperBounds()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := gub[0], ub[0]; got != want {
		t.Errorf("Expected ub='%f', got='%f'", want, got)
	}
	if got, want := gub[1], ub[1]; got != want {
		t.Errorf("Expected ub='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetUpperBounds1(t *testing.T) {
	o, err := NewNLopt(LD_MMA, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	ub := -1.
	err = o.SetUpperBounds1(ub)
	if err != nil {
		t.Fatal(err)
	}
	gub, err := o.GetUpperBounds()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := gub[0], ub; got != want {
		t.Errorf("Expected ub='%f', got='%f'", want, got)
	}
	if got, want := gub[1], ub; got != want {
		t.Errorf("Expected ub='%f', got='%f'", want, got)
	}
}

func TestNLopt_RemoveInequalityConstraints(t *testing.T) {
	o, err := NewNLopt(LD_MMA, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.RemoveInequalityConstraints()
	if err != nil {
		t.Fatal(err)
	}
}

func TestNLopt_AddInequalityConstraint(t *testing.T) {
	o, err := NewNLopt(LD_MMA, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	fc := func(x []float64, gradient []float64) float64 {
		return math.Inf(-1)
	}
	err = o.AddInequalityConstraint(fc, 0.)
	if err != nil {
		t.Fatal(err)
	}
	err = o.RemoveInequalityConstraints()
	if err != nil {
		t.Fatal(err)
	}
}

func TestNLopt_AddInequalityMConstraint(t *testing.T) {
	o, err := NewNLopt(LD_MMA, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	fc := func(result []float64, x []float64, gradient []float64) {
		return
	}
	err = o.AddInequalityMConstraint(fc, []float64{0.})
	if err != nil {
		t.Fatal(err)
	}
	err = o.RemoveInequalityConstraints()
	if err != nil {
		t.Fatal(err)
	}
}

func TestNLopt_RemoveEqualityConstraints(t *testing.T) {
	o, err := NewNLopt(LD_MMA, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.RemoveEqualityConstraints()
	if err != nil {
		t.Fatal(err)
	}
}

func TestNLopt_AddEqualityConstraint(t *testing.T) {
	o, err := NewNLopt(LD_SLSQP, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	fc := func(x, gradient []float64) float64 {
		return math.Inf(-1)
	}
	err = o.AddEqualityConstraint(fc, 0.)
	if err != nil {
		t.Fatal(err)
	}
	err = o.RemoveEqualityConstraints()
	if err != nil {
		t.Fatal(err)
	}
}

func TestNLopt_AddEqualityMConstraint(t *testing.T) {
	o, err := NewNLopt(LD_SLSQP, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	fc := func(result, x, gradient []float64) {
		return
	}
	err = o.AddEqualityMConstraint(fc, []float64{0.})
	if err != nil {
		t.Fatal(err)
	}
	err = o.RemoveEqualityConstraints()
	if err != nil {
		t.Fatal(err)
	}
}

// stopping criteria:

func TestNLopt_SetStopVal(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetStopVal(2.12)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetStopVal(), 2.12; got != want {
		t.Errorf("Expected stopval='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetFtolRel(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetFtolRel(1.73)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetFtolRel(), 1.73; got != want {
		t.Errorf("Expected tol='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetFtolAbs(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetFtolAbs(11.32)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetFtolAbs(), 11.32; got != want {
		t.Errorf("Expected tol='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetXtolRel(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetXtolRel(3.45)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetXtolRel(), 3.45; got != want {
		t.Errorf("Expected tol='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetXtolAbs1(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetXtolAbs1(753.1)
	if err != nil {
		t.Fatal(err)
	}
	v, err := o.GetXtolAbs()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := v[0], 753.1; got != want {
		t.Errorf("Expected tol='%f', got='%f'", want, got)
	}
	if got, want := v[1], 753.1; got != want {
		t.Errorf("Expected tol='%f', got='%f'", want, got)
	}
	if got, want := v[2], 753.1; got != want {
		t.Errorf("Expected tol='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetXtolAbs(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	tol := []float64{356.79, 987.654}
	err = o.SetXtolAbs(tol)
	if err != nil {
		t.Fatal(err)
	}
	v, err := o.GetXtolAbs()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := v[0], 356.79; got != want {
		t.Errorf("Expected tol='%f', got='%f'", want, got)
	}
	if got, want := v[1], 987.654; got != want {
		t.Errorf("Expected tol='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetMaxEval(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetMaxEval(12)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetMaxEval(), 12; got != want {
		t.Errorf("Expected maxeval='%d', got='%d'", want, got)
	}
}

func TestNLopt_SetMaxTime(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetMaxTime(11.5)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetMaxTime(), 11.5; got != want {
		t.Errorf("Expected maxtime='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetForceStop(t *testing.T) {
	o, err := NewNLopt(LD_AUGLAG, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetForceStop(1)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetForceStop(), 1; got != want {
		t.Errorf("Expected val='%d', got='%d'", want, got)
	}
}

// more algorithm-specific parameters

func TestNLopt_SetLocalOptimizer(t *testing.T) {
	o, err := NewNLopt(LD_TNEWTON_RESTART, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()

	lo, err := NewNLopt(GN_DIRECT_L_RAND, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer lo.Destroy()

	err = o.SetLocalOptimizer(lo)
	if err != nil {
		t.Fatal(err)
	}
}

func TestNLopt_SetPopulation(t *testing.T) {
	o, err := NewNLopt(GN_MLSL, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetPopulation(uint(237))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetPopulation(), uint(237); got != want {
		t.Errorf("Expected population='%d', got='%d'", want, got)
	}
}

func TestNLopt_SetVectorStorage(t *testing.T) {
	o, err := NewNLopt(LD_TNEWTON_RESTART, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	err = o.SetVectorStorage(uint(123))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := o.GetVectorStorage(), uint(123); got != want {
		t.Errorf("Expected vector storage='%d', got='%d'", want, got)
	}
}

func TestNLopt_GetInitialStep(t *testing.T) {
	o, err := NewNLopt(GN_ESCH, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	x, dx, err := o.GetInitialStep()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := x[0], 0.; got != want {
		t.Errorf("Expected x='%f', got='%f'", want, got)
	}
	if got, want := dx[0], 1.; got != want {
		t.Errorf("Expected dx='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetDefaultInitialStep(t *testing.T) {
	o, err := NewNLopt(GN_ESCH, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	dis := []float64{2.1}
	err = o.SetDefaultInitialStep(dis)
	if err != nil {
		t.Fatal(err)
	}
	x, dx, err := o.GetInitialStep()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := x[0], 0.; got != want {
		t.Errorf("Expected x='%f', got='%f'", want, got)
	}
	if got, want := dx[0], dis[0]; got != want {
		t.Errorf("Expected dx='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetInitialStep(t *testing.T) {
	o, err := NewNLopt(GN_ESCH, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	is := []float64{10., 20.}
	err = o.SetInitialStep(is)
	if err != nil {
		t.Fatal(err)
	}
	x, dx, err := o.GetInitialStep()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := x[0], 0.; got != want {
		t.Errorf("Expected x='%f', got='%f'", want, got)
	}
	if got, want := dx[0], is[0]; got != want {
		t.Errorf("Expected dx='%f', got='%f'", want, got)
	}
	if got, want := dx[1], is[1]; got != want {
		t.Errorf("Expected dx='%f', got='%f'", want, got)
	}
}

func TestNLopt_SetInitialStep1(t *testing.T) {
	o, err := NewNLopt(GN_ESCH, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Destroy()
	is := 10.0
	err = o.SetInitialStep1(is)
	if err != nil {
		t.Fatal(err)
	}
	x, dx, err := o.GetInitialStep()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := x[0], 0.; got != want {
		t.Errorf("Expected x='%f', got='%f'", want, got)
	}
	if got, want := dx[0], is; got != want {
		t.Errorf("Expected dx='%f', got='%f'", want, got)
	}
	if got, want := dx[1], is; got != want {
		t.Errorf("Expected dx='%f', got='%f'", want, got)
	}
}
