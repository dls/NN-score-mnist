Trains a net using sum squared error rather than cross entropy.

$ diff 2_train_and_run.jl ../2_train_and_run.jl
114c114
<     l += (model(x) .- y) .^ 2
---
>     l += logitcrossentropy(model(x), y)
