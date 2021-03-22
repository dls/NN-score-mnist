using Serialization
using Optim, LineSearches
using NNlib
using DataFrames, Query

h(x) = round(x * 100) / 100

DATA = "2_scored_test_data.dat"
if !(@isdefined(test1))
  (test1, test2) = Serialization.deserialize(DATA)
end

claimed_category(x) = @filter(_.classified_category == x)
score_in_bounds(lower, upper) = @filter(lower < _.score <= upper)
true_positive() = @filter(_.correct_category == _.classified_category)
false_positive() = @filter(_.correct_category != _.classified_category)
count(df) = size(df |> DataFrame)[1]

function HISTOGRAM_BINNING__thetas_and_error_for_cuttoffs(df, cuttoffs_; force_theta_monotonicity=false)
  cuttoffs = sort(copy(cuttoffs_))
  thetas = Float64[]
  theta_min_bound = 0.0
  err = 0

  function sum_errs(r, theta)
    err = 0
    for e=eachrow(r)
      if e.correct_category == e.classified_category
        err += 1 - theta
      else
        err += theta
      end
    end
    return err
  end

  function optimize_range(lower, upper)
    r = df |> score_in_bounds(lower, upper) |> DataFrame
    theta = count(r |> true_positive()) / (count(r) + 0.00001)
    if force_theta_monotonicity
      theta = max(theta_min_bound, theta)
      theta_min_bound = theta
    end
    err += sum_errs(r, theta)
    push!(thetas, theta)
  end

  optimize_range(0, cuttoffs[1])
  for i=2:length(cuttoffs)
    optimize_range(cuttoffs[i-1], cuttoffs[i])
  end
  optimize_range(cuttoffs[end], 10)

  return (thetas, err)
end

function ISOTONIC__thetas_and_error_for_cuttoffs(df, cuttoffs_)
  (thetas, err) = HISTOGRAM_BINNING__thetas_and_error_for_cuttoffs(df, cuttoffs_; force_theta_monotonicity=true)

  # "constrains" theta[i] to be <= theta[i+1]
  constraints = map(e -> max(0, 1e5 * (e[1] - e[2])),
                    zip(thetas[1:end-1], thetas[2:end]))
  err += sum(constraints)

  # "constrains" a[1] to be >= 0
  err += max(0, 1e5 * -minimum(cuttoffs_))
  # "constrains" a[end] to be <= 10
  err += max(0, 1e5 * (maximum(cuttoffs_) - 10))

  return (thetas, err)
end

function min_by(f, xs)
  min_v = Inf
  min_i = -1

  for i=1:length(xs)
    v = f(xs[i])
    if v < min_v
      min_v = v
      min_i = i
    end
  end

  return xs[min_i]
end

n_optimize(n, f, start_f) = min_by(e -> e.minimum,
                                   [optimize(f, start_f(), NelderMead(), Optim.Options(x_tol = 1e-3)) for i=1:n])

function run_iso(df, cat)
  df = df |> claimed_category(cat) |> DataFrame
  c = n_optimize(10, x -> ISOTONIC__thetas_and_error_for_cuttoffs(df, x)[2],
                 () -> rand(4) * 10)

  @show h.(sort(c.minimizer))
  @show sort(c.minimizer)
  t, e = ISOTONIC__thetas_and_error_for_cuttoffs(df, c.minimizer)
  @show h.(t)
  @show e
  @show count(df)
  @show e / count(df)
end

function main()
  for i=0:9
    println()
    println("Running on test1, category $i")
    run_iso(test1, i)
  end
end

main()
