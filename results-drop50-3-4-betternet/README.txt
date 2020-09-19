Runs the system dropping 50% of categories 3 and 4, with the model enlarged.

$ diff 2_train_and_run.jl ../2_train_and_run.jl
106,108c106,107
<  	  Dense(prod(imgsize), 256, relu),
<     Dense(256, 64),
<     Dense(64, nclasses),
---
>  	  Dense(prod(imgsize), 32, relu),
>     Dense(32, nclasses),
