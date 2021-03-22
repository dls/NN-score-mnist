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

function scored_pct(pos, neg, lower, upper)
  yeses = 0
  nos = 0

  seen = 0
  for cat_group in cat(pos, neg; dims=1)
    for e in cat_group
      seen += 1
      m = maximum(e.raw_output[1,:])

      if lower <= m <= upper
        i = indexin(m, e.raw_output[1,:])[1]
        if e.correct_category + 1 == i
          yeses += 1
        else
          nos += 1
        end
      end
    end
  end
  @show seen

  return yeses / (yeses + nos + 0.0001)
end

function main()
  DATA = "2_scored_test_data.dat"
  ((test1_pos, test1_neg), (test2_pos, test2_neg)) = Serialization.deserialize(DATA)

  h(x) = round(100 * x) / 100.0

  function univarite_score(i, f)
    err = 0

    for cat_group in cat(test1_pos, test1_neg; dims=1)
      for e in cat_group
        m = maximum(e.raw_output[i,:])
        idx = indexin(m, e.raw_output[i,:])[1]

        r = f(m)
        a = e.correct_category + 1 == idx ? 1.0 : 0.0
        err += (r - a) ^ 2
      end
    end

    return err
  end

  function isotonic(cuttoffs, thetas)
    cuttoffs = sort(cuttoffs)
    @show h.(cuttoffs)
    # @show thetas
    return function f(x)
      for i=1:9
        if x < cuttoffs[i]
          return thetas[i]
        end
      end
      return thetas[10]
    end
  end

  validate(i, cuttoffs, thetas) = univarite_score(i, isotonic(cuttoffs, thetas))
  validate_i(i) = (temp) -> validate(i, temp[1:9], temp[10:19])

  N = 1
  optimal_iso = zeros(N, 19)
  starting_hypothesis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  for i=1:10
    starting_hypothesis[i] = rand()
  end

  for i in 1:N
    result = optimize(validate_i(i), starting_hypothesis, NelderMead(), Optim.Options(x_tol=1e-3))
    optimal_iso[i,:] = Optim.minimizer(result)
  end

  @show h.(optimal_iso)

  return optimal_iso
end

RES = main()
