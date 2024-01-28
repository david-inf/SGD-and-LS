# Optimization-Project

## Contents

- [ ] Introduction
- The logistic regression problem
- [ ] Implementation

## To do

Python
- [x] conda local environment with:
    - numpy, scipy, pandas, scikit-learn
- [x] splittare nei minibatch

Logistic regression - theory
- binary classification
- the problem admits a unique optimal solution

Logistic regression - python
- python class? like in scikit?
- class
    - init: hyperparameters
    - get objective function
    - get gradient

Metrics/diagnostic
- training loss against epochs
    - su un numero fissato di epoche: while k<k*
- training loss against iterations
    - per iterations si intende quelle interne ad una epoca
    - le iterations sono fissate una volta assegnata la dimensione del minibatch
- benchmark: scipy.optimize.minimize
    - fun e jac devono avere un solo argomento, va ripensato come chiamarla

Scoperte
- miniGD con passo costante è non monotono, o almeno per certi valori dipendenti dal dataset
    - deve essere fatto un fine-tuning sul dataset
    - è non monotono proprio perché senza il fine-tuning il passo costante non è ottimo

Altro
- speed up execution
    - numba, cuda
