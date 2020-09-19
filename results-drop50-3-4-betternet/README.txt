Runs the system dropping 50% of categories 3 and 4, with the model enlarged.

$ diff 2_train_and_run.jl ../2_train_and_run.jl
15c15
<   epochs::Int = 50        # number of epochs
---
>   epochs::Int = 10        # number of epochs
106,108c106,107
<  	  Dense(prod(imgsize), 256, leakyrelu),
<     Dense(256, 128, leakyrelu),
<     Dense(128, nclasses),
---
>  	  Dense(prod(imgsize), 32, relu),
>     Dense(32, nclasses),
