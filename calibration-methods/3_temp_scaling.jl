using Serialization
using Optim, LineSearches
using NNlib

struct RunResult
  raw_output :: Array{Float64, 2}
  output :: Array{Float64, 1}
  max_output_i :: Int64
  score :: Float64
  correct_category :: Int64
end

function main()
  DATA = "2_scored_test_data.dat"
  ((test1_pos, test1_neg), (test2_pos, test2_neg)) = Serialization.deserialize(DATA)

  function score(i, f)
    err = 0
    z = zeros(10)
    function onehotted(i)
      z *= 0.0
      z[i] = 1.0
      return z
    end

    for cat_group in cat(test1_pos, test1_neg; dims=1)
      for e in cat_group
        r = f(e.raw_output[i,:])
        a = onehotted(e.correct_category + 1)
        err += sum((r .- a) .^ 2)
      end
    end

    return err
  end

  validate(i, temp) = score(i, (x) -> softmax(x ./ temp))
  validate_i(i) = (temp) -> validate(i, temp[1])

  optimal_temps = zeros(10)

  for i in 1:10
    result = optimize(validate_i(i), [1.0], BFGS())
    # @show result
    optimal_temps[i] = Optim.minimizer(result)[1]
  end

  @show optimal_temps

  err_diffs = [validate(i, 1.0) - validate(i, optimal_temps[i]) for i=1:10]

  @show [validate(i, 1.0) for i=1:10]
  @show [validate(i, optimal_temps[i]) for i=1:10]

  @show err_diffs

  return optimal_temps
end

RES = main()
