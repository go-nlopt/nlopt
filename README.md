A NLopt implementation for Go
======

A package to provide functionality of object-oriented C-API of [NLopt](http://ab-initio.mit.edu/wiki/index.php/Main_Page) 
for the Go programming language (http://golang.org). This provides a wrapper 
using cgo to a c-based implementation.


## Status

*Beta*

[![Build Status](https://travis-ci.org/go-nlopt/nlopt.svg?branch=master)](https://travis-ci.org/go-nlopt/nlopt) [![Coverage Status](https://coveralls.io/repos/github/go-nlopt/nlopt/badge.svg?branch=master)](https://coveralls.io/github/go-nlopt/nlopt?branch=master) [![GoDoc](https://godoc.org/gopkg.in/nlopt.v0?status.svg)](https://godoc.org/gopkg.in/nlopt.v0)


## Installation

- On Fedora

~~~
dnf -y install libnlopt-devel
~~~

- On Ubuntu

~~~
apt-get install libnlopt-dev
~~~

- or, install NLopt library on any Unix-like system (GNU/Linux is fine) with a 
  C compiler, using the standard procedure:

~~~
curl -O http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz && tar xzvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
./configure --enable-shared && make && sudo make install
~~~

- On Windows download binary packages at [NLopt on Windows](http://ab-initio.mit.edu/wiki/index.php/NLopt_on_Windows)

Then install `nlopt` package

~~~
go get gopkg.in/nlopt.v0
[CGO_LDFLAGS="-L/path/to/NLopt -lnlopt"] go install gopkg.in/nlopt.v0
~~~


## Example

Adaptation of nonlinearly constrained problem from [NLopt Tutorial](http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial)

~~~go
package main

import (
        "fmt"
        "gopkg.in/nlopt.v0"
        "math"
)

func main() {
        opt, err := nlopt.NewNLopt(nlopt.LD_MMA, 2)
        if err != nil {
                panic(err)
        }
        defer opt.Destroy()

        opt.SetLowerBounds([]float64{math.Inf(-1), 0.})

        var evals int
        myfunc := func(x []float64, gradient []float64) float64 {
                evals++
                if len(gradient) > 0 {
                        gradient[0] = 0.0
                        gradient[1] = 0.5 / math.Sqrt(x[1])
                }
                return math.Sqrt(x[1])
        }
        
        myconstraint := func(x []float64, gradient []float64, a, b float64) float64 {
                if len(gradient) > 0 {
                        gradient[0] = 3*a* math.Pow(a*x[0]+b, 2.)
                        gradient[1] = -1.0
                }
                return math.Pow(a*x[0]+b, 3) - x[1]
        }

        opt.SetMinObjective(myfunc)
        opt.AddInequalityConstraint(func(x []float64, gradient []float64) float64 { return myconstraint(x, gradient, 2., 0.)}, 1e-8)
        opt.AddInequalityConstraint(func(x []float64, gradient []float64) float64 { return myconstraint(x, gradient, -1., 1.)}, 1e-8)
        opt.SetXtolRel(1e-4)

        x := []float64{1.234, 5.678}
        xopt, minf, err := opt.Optimize(x)
        if err != nil {
                panic(err)
        }
        fmt.Printf("found minimum after %d evaluations at f(%g,%g) = %0.10g\n", evals, xopt[0], xopt[1], minf)
}
~~~

## License

MIT - see LICENSE for more details.