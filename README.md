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
    - {-1,1} oppure {0,1}??? is risultati ovviamente cambiano
    - se {-1,1} va cambiata la sigmoide??
- the problem admits a unique optimal solution
- [ ] devo dividere per N??
- [ ] dividere per 2 il regolarizzatore??

Ottimizzazione:
- criteri di arresto
    - qualcosa per metodi non monotoni?

Logistic regression - python
- python class? like in scikit?
- class
    - init: hyperparameters
    - get objective function
    - get gradient

Metrics/diagnostic
- [x] training loss against epochs
    - su un numero fissato di epoche: while k<k*
- [x] timer per esecuzione
- training loss against iterations
    - per iterations si intende quelle interne ad una epoca
    - le iterations sono fissate una volta assegnata la dimensione del minibatch
- [x] benchmark: scipy.optimize.minimize
    - fun e jac devono avere un solo argomento, va ripensato come chiamarla -> gliene posso passare altri con args=()

Scoperte
- miniGD con passo costante è non monotono, o almeno per certi valori dipendenti dal dataset
    - deve essere fatto un fine-tuning sul dataset
    - è non monotono proprio perché senza il fine-tuning il passo costante non è ottimo

Altro
- speed up execution
    - numba, cuda
    - pytorch
