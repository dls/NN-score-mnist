using Serialization
using Optim, LineSearches
using NNlib
using DataFrames, Query
import Base.show

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

# TODO: softmaxlike -> vector
function _score_elementwise_softmaxlike_f(df, f)
  err = 0
  for e=eachrow(df)
    res = f(e)
    category_val = res[e.correct_category + 1]
    if e.correct_category == e.classified_category
      err += 1 - category_val
    else
      err += category_val
    end
  end
  return err
end

# TODO: _score_elementwise_scalar_f

function TSCALE_COMBINED__error_for_temp(df, t)
  tscale(e) = NNlib.softmax(sum(e.net_output, dims=1) ./ t)
  return _score_elementwise_softmaxlike_f(df, tscale)
end

function TSCALE_SEPARATE__error_for_temp(df, i, t)
  tscale(e) = NNlib.softmax(e.net_output[i,:] ./ t)
  return _score_elementwise_softmaxlike_f(df, tscale)
end

function PLATT_COMBINED__err(df, a, b)
  zs = zeros(10)
  function platt(e)
    zs *= 0
    zs[e.classified_category + 1] = NNlib.sigmoid(a * maximum(sum(e.net_output, dims=1)) + b)
    return zs
  end
  return _score_elementwise_softmaxlike_f(df, platt)
end

function PLATT_SEPARATE__err(df, i, a, b)
  zs = zeros(10)
  function platt(e)
    zs *= 0
    zs[e.classified_category + 1] = NNlib.sigmoid(a * maximum(e.net_output[i, :]) + b)
    return zs
  end
  return _score_elementwise_softmaxlike_f(df, platt)
end

function MATRIX_COMBINED_err(df, w, b)
  matrix_scale(e) = NNlib.sigmoid.(sum(e.net_output, dims=1) * w .+ b')
  return _score_elementwise_softmaxlike_f(df, matrix_scale)
end

function MATRIX_SEPARATE_err(df, i, w, b)
  matrix_scale(e) = NNlib.sigmoid(e.net_output[i, :] * w + b)
  return _score_elementwise_softmaxlike_f(df, matrix_scale)
end

function min_by(f, xs)
  min_v = Inf
  min_i = -1

  for i=1:length(xs)
    v = f(xs[i])
    @show v
    if v < min_v
      min_v = v
      min_i = i
    end
  end

  return xs[min_i]
end

n_optimize(n, f, start_f; optimizer=NelderMead()) = min_by(e -> e.minimum, [optimize(f, start_f(), optimizer, Optim.Options(x_tol = 1e-3)) for i=1:n])

function run_iso(df)
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

function cumulative_err(source, sink)
  err = 0
  for e=eachrow(sink)
    items_greater_than = source |> score_in_bounds(e.score, 10) |> DataFrame
    estimate = count(items_greater_than |> true_positive()) / (count(items_greater_than) + 0.0001)
    if e.correct_category == e.classified_category
      err += 1 - estimate
    else
      err += estimate
    end
  end
  return err
end

# ** adrian score: sum([softmax(net_output[i]) for i=1:10])

# # number of variables: length(category)
# ** t-scaling hypothesis: softmax(sum([net_output[i] for i=1:10])/t)

# # number of variables: length(category) * length(nets)
# ** peter prefers: sum([softmax(net_output[i]/t) for i=1:10])/10
run_tscale_combined(df) = n_optimize(10, x -> TSCALE_COMBINED__error_for_temp(df, x[1]), () -> rand(1); optimizer=Newton())
run_tscale_separate(df) = n_optimize(10, x -> TSCALE_SEPARATE__error_for_temp(df, x[1]), () -> rand(1); optimizer=Newton())

function run_matrix_combined(df)
  function run(x)
    w = reshape(x[1:100], (10, 10))
    b = x[101:110]
    return MATRIX_COMBINED_err(df, w, b)
  end
  n_optimize(10, run, () -> rand(110))
end

function main()
  for i=0:9
    df1 = test1 |> claimed_category(i) |> DataFrame
    println()
    println("Running on test1, category $i")
    r = run_tscale_combined(df1)
    # @show r
    @show r.minimizer, r.minimum, r.minimum / count(df1)

    df2 = test2 |> claimed_category(i) |> DataFrame
    err2 = TSCALE_COMBINED__error_for_temp(df2, r.minimizer[1])
    @show err2, err2 / count(df2)
  end
end

function main2()
  for i=0:9
    println()
    println("Running on category $i")

    df1 = test1 |> claimed_category(i) |> DataFrame
    df2 = test2 |> claimed_category(i) |> DataFrame

    err1 = cumulative_err(df1, df2)
    @show err1, err1 / count(df1)

    err2 = cumulative_err(df2, df1)
    @show err2, err2 / count(df2)
  end
end

# main()

mutable struct Segmentation
  bins :: Array{Float64, 1}
  thetas :: Array{Float64, 1}
  p :: Float64
end

function Base.show(io::IO, s::Segmentation)
  println(io, "p $(s.p):")
  println(io, "\tbins: $(s.bins)")
  println(io, "\tthetas: $(s.thetas)")
end

function bbq(df, target_df; SAMPLES_COUNT=100, BIN_COUNT=10)
  function theta_for_range(lower, upper)
    r = df |> score_in_bounds(lower, upper) |> DataFrame
    return count(r |> true_positive()) / (count(r) + 0.00001)
  end

  function thetas_for_bins(bins)
    thetas = zeros(length(bins)+1)
    thetas[1] = theta_for_range(0.0, bins[1])
    for i=2:length(bins)
      thetas[i] = theta_for_range(bins[i-1], bins[i])
    end
    thetas[end] = theta_for_range(bins[end], 10.0)
    return thetas
  end

  function theta_for_row(e, bins, thetas)
    for i=1:length(bins)
      if e.score < bins[i]
        return thetas[i]
      end
    end
    return thetas[end]
  end

  function p_of_s(bins, thetas)
    p = 0.0
    for e=eachrow(df)
      t = theta_for_row(e, bins, thetas)
      if e.correct_category == e.classified_category
        p += log(t)
      else
        p += log(1 - t)
      end
    end
    return exp(p)
  end

  function mk_s()
    bins = sort(rand(BIN_COUNT)) * 10
    thetas = thetas_for_bins(bins)
    p = p_of_s(bins, thetas)
    return Segmentation(bins, thetas, p)
  end

  segs = [mk_s() for i=1:SAMPLES_COUNT]
  normalizatoin_factor = sum([s.p for s=segs])
  for e=segs
    e.p /= normalizatoin_factor
  end

  err = 0.0
  for e=eachrow(target_df)
    estimate = 0.0
    for s=segs
      estimate += s.p * theta_for_row(e, s.bins, s.thetas)
    end

    if e.correct_category == e.classified_category
      err += 1 - estimate
    else
      err += estimate
    end
  end
  return err
end

function main3(samples_count, bin_count)
  println("Running BBQ with $samples_count samples, $bin_count bins")
  for i=0:9
    print("  category $i: ")

    df1 = test1 |> claimed_category(i) |> DataFrame
    df2 = test2 |> claimed_category(i) |> DataFrame

    err1 = bbq(df1, df2; SAMPLES_COUNT=samples_count, BIN_COUNT=bin_count) / count(df2)
    err2 = bbq(df2, df1; SAMPLES_COUNT=samples_count, BIN_COUNT=bin_count) / count(df1)

    println("$(h(100*err1))%\t$(h(100*err2))%")
  end
end

#main3()

# TODO:
# - maybe stare at early results?
# - decide how to officially judge
