#!/usr/bin/env python

import numpy, numpy as np
import theano.tensor as TT, theano
from pylearn2.optimization.ncg import ncg
from python_cg import *
import time
import cPickle


# Global parameters
floatX      = theano.config.floatX
profile     = True
pydot_print = True

def test_ncg_1():

    rng = numpy.random.RandomState(232)

    all_vals = numpy.asarray(
        rng.uniform(size=(5*5+5,)),
        dtype=theano.config.floatX)

    idx  = 0
    vW0  = all_vals[idx : idx + 25].reshape((5,5))
    idx  = 25
    vb0  = all_vals[idx : idx + 5].reshape((5,))

    vx  = numpy.asarray(
        rng.uniform(size=(20,5)), dtype=theano.config.floatX)
    vy  = numpy.asarray(
        rng.uniform(size=(20,5)), dtype=theano.config.floatX)

    W0 = theano.shared(vW0, 'W0')
    b0 = theano.shared(vb0 ,'b0')

    x = theano.shared(vx, 'x')
    y = theano.shared(vy, 'y')

    def f(W0,b0,x, y):
        p1 = TT.nnet.sigmoid(TT.dot(x,W0) + b0)
        # Strange function !?
        return ((p1-y)**2).sum()

    print 'Executing ncg'
    print '>>> Generating Graph'
    t0 = time.time()
    answers = ncg(f, [W0,b0], [x,y], maxiter = 60,
                      profile = profile)
    tf = time.time() - t0
    print 'It took', tf, 'sec'
    print '>>> Compiling graph'
    t0 = time.time()
    func = theano.function([], answers, profile = profile)
    tf = time.time() - t0
    print 'It took', tf, 'sec'
    if pydot_print:
        print '>>> Plotting graph'
        #theano.plot.Ipydotprint(func)
    print 'Optimizing'
    t_th = 0
    t_py = 0
    for k in xrange(20):
        t0 = time.time()
        th_rval = func()[0]
        t_th += time.time() - t0

    print '-------- NOW SCIPY RESULTS ------'
    allw = TT.vector('all')
    idx  = 0
    W0   = allw[:25].reshape((5,5))
    idx  = 25
    b0   = allw[idx:idx+5].reshape((5,))
    out    = f(W0,b0,x,y)
    func   = theano.function([allw], out)
    gall   = TT.grad(out, allw)
    fprime = theano.function([allw], gall)

    if pydot_print:
        theano.printing.pydotprint(func, 't1_f.png', with_ids = True,
                                   high_contrast = True)
        theano.printing.pydotprint(fprime, 't1_fprime.png', with_ids = True,
                                   high_contrast = True)

    # FIRST RUN with full_output to get an idea of how many steps where done
    t0 = time.time()
    rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                  maxiter = 60,
                                   full_output = 1)[1]

    t_py += time.time() - t0
    # rest runs with full_output 0
    for k in xrange(19):
        t0 = time.time()
        rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                      maxiter = 60,
                                       full_output = 1,
                                       disp = 0 )[1]

        t_py += time.time() - t0

    print 'THEANO output :: ',th_rval
    print 'NUMPY  output :: ',rval
    print
    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy  ---------> time %e'% t_py


def test_ncg_2():

    rng = numpy.random.RandomState(232)

    all_vals = numpy.asarray(
        rng.uniform(size=(500*500,)),
        dtype=theano.config.floatX)

    idx  = 0
    vW0  = all_vals.reshape((500,500))

    vx  = numpy.asarray(
        rng.uniform(size=(2000,500)), dtype=theano.config.floatX)
    vy  = numpy.asarray(
        rng.uniform(size=(2000,500)), dtype=theano.config.floatX)

    W0 = theano.shared(vW0, 'W0')
    #W0  = TT.specify_shape(_W0, vW0.shape)
    #W0.name = 'W0'

    x = theano.shared(vx, 'x')
    #x  = TT.specify_shape(_x, vx.shape)
    #x.name = 'x'
    y = theano.shared(vy, 'y')
    #y  = TT.specify_shape(_y, vy.shape)
    #y.name = 'y'

    def f(W0):
        return ((TT.dot(x,W0) - y)**2).mean().mean()
        #return ((TT.dot(x,W0) - y)**2).mean().mean() + abs(x).mean().mean()

    print 'Executing ncg'
    print '>>> Generating Graph'
    t0 = time.time()
    answers = ncg(f, [W0], [], maxiter = 6,
                      profile = profile)
    tf = time.time() - t0

    print 'It took', tf, 'sec'
    print '>>> Compiling graph'
    t0 = time.time()
    func = theano.function([], answers, profile = profile,
                           name = 'test_fmincg_2',
                          mode = theano.Mode(linker='cvm'))
    tf = time.time() - t0
    print 'It took', tf, 'sec'
    if pydot_print:
        print '>>> Plotting graph'
    theano.printing.pydotprint(func,'t2_fmin_cg.png',
                           with_ids = True,
                           high_contrast = True,
                           scan_graphs = True)
    print 'Optimizing'
    t_th = 0
    t_py = 0
    for k in xrange(1):
        t0 = time.time()
        th_rval = func()[0]
        t_th += time.time() - t0

    print '-------- NOW SCIPY RESULTS ------'
    allw = TT.vector('all')
    #allw  = TT.specify_shape(_allw, all_vals.shape)
    idx  = 0
    W0   = allw.reshape((500,500))
    out    = f(W0)
    func   = theano.function([allw], out)
    gall   = TT.grad(out, allw)
    fprime = theano.function([allw], gall)

    if pydot_print:
        theano.printing.pydotprint(func,  't2_f.png', with_ids = True,
                                   high_contrast = True)
        theano.printing.pydotprint(fprime,'t2_fprime.png', with_ids = True,
                                   high_contrast = True)

    # FIRST RUN with full_output to get an idea of how many steps where done
    t0 = time.time()
    rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                  maxiter = 6,
                                   full_output = 1,
                                    disp = 1)[1]

    t_py += time.time() - t0
    # rest runs with full_output 0
    '''
    for k in xrange(1):
        t0 = time.time()
        rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                      maxiter = 6,
                                       full_output = 1,
                                       disp = 0 )[1]

        t_py += time.time() - t0
    '''
    print 'THEANO output :: ',th_rval
    print 'NUMPY  output :: ',rval
    print
    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy  ---------> time %e'% t_py


def test_ncg_3():

    rng = numpy.random.RandomState(232)

    all_vals = numpy.asarray(
        rng.uniform(size=(500*500,)),
        dtype=theano.config.floatX)

    idx  = 0
    vW0  = all_vals.reshape((500,500))

    vu  = numpy.asarray(
        rng.uniform(size=(2000,500)), dtype=theano.config.floatX)
    vx  = numpy.asarray(
        rng.uniform(size=(2000,500)), dtype=theano.config.floatX)
    vy  = numpy.asarray(
        rng.uniform(size=(2000,500)), dtype=theano.config.floatX)

    vW1 = numpy.asarray(
        rng.uniform(size=(500,500)), dtype = theano.config.floatX)

    W1 = theano.shared(vW1, 'W1')

    W0 = theano.shared(vW0, 'W0')
    x = theano.shared(vx, 'x')
    u = theano.shared(vu, 'u')
    y = theano.shared(vy, 'y')

    def f(W0):
        return TT.mean((TT.tanh(TT.dot(TT.tanh(TT.dot(x,W1)+u),W0))
                        - y)**2)


    print 'Executing ncg'
    print '>>> Generating Graph'
    t0 = time.time()
    answers = ncg(f, [W0], [], maxiter = 60,
                      profile = profile)
    tf = time.time() - t0

    print 'It took', tf, 'sec'
    print '>>> Compiling graph'
    t0 = time.time()
    func = theano.function([], answers, profile = profile,
                           name = 'test_fmincg_2',
                          mode = theano.Mode(linker='cvm'))

    tf = time.time() - t0
    print 'It took', tf, 'sec'
    if pydot_print:
        print '>>> Plotting graph'
    theano.printing.pydotprint(func,'t2_fmin_cg.png',
                           with_ids = True,
                           high_contrast = True,
                           scan_graphs = True)
    print 'Optimizing'
    t_th = 0
    t_py = 0
    for k in xrange(1):
        t0 = time.time()
        th_rval = func()[0]
        t_th += time.time() - t0

    print '-------- NOW SCIPY RESULTS ------'
    allw = TT.vector('all')
    idx  = 0
    W0   = allw.reshape((500,500))
    out    = f(W0)
    func   = theano.function([allw], out)
    gall   = TT.grad(out, allw)
    fprime = theano.function([allw], gall)

    if pydot_print:
        theano.printing.pydotprint(func,  't2_f.png', with_ids = True,
                                   high_contrast = True)
        theano.printing.pydotprint(fprime,'t2_fprime.png', with_ids = True,
                                   high_contrast = True)

    # FIRST RUN with full_output to get an idea of how many steps where done
    t0 = time.time()
    rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                  maxiter = 60,
                                   full_output = 1,
                                    disp = 1)[1]

    t_py += time.time() - t0

    print 'THEANO output :: ',th_rval
    print 'NUMPY  output :: ',rval
    print
    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy  ---------> time %e'% t_py


if __name__ == '__main__':
    #print 'Test 1:'
    #test_ncg_1()
    print
    print
    print
    print 'Test 2:'
    test_ncg_2()
