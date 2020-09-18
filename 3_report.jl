using DataFrames
using Gadfly
using Serialization

import Cairo, Fontconfig


struct RunResult
  output :: Array{Float32, 1}
  max_output_i :: Int64
  score :: Float32
  correct_category :: Int64
end

function percent_greater_than(x, pos, neg)
  pos_len = length(filter(e -> e.score > x, pos)) + 1e-30 # prevent divide by 0
  neg_len = length(filter(e -> e.score > x, neg))
  return pos_len / (pos_len + neg_len)
end

function percent_less_than(x, pos, neg)
  pos_len = length(filter(e -> e.score < x, pos))
  neg_len = length(filter(e -> e.score < x, neg)) + 1e-30 # prevent divide by 0
  return pos_len / (pos_len + neg_len)
end

function confidence_plus_softmax(f, set_id, pos1, neg1)
  println(f, "dataset, predicted category, actual category, true_positive, score, gt confidence, lt confidence")
  for cat=1:10
    for e=sort(vcat(pos1[cat], neg1[cat]), by = e -> e.score)
      gt = percent_greater_than(e.score, pos1[cat], neg1[cat])
      lt = percent_less_than(e.score, pos1[cat], neg1[cat])
      correct = e.max_output_i == e.correct_category ? 1 : 0
      println(f, "$set_id, $cat, $(e.correct_category + 1), $correct, $(e.score), $gt, $lt")
    end
  end
end

function ks_greater_than(pos1, neg1, pos2, neg2)
  ks_result = Array{Float64,1}()
  for cat in 1:10
    ks = 0
    for score in 0:0.1:10
      p = percent_greater_than(score, pos1[cat], neg1[cat])
      q = percent_greater_than(score, pos2[cat], neg2[cat])
      ks = max(abs(p - q), ks)
    end
    push!(ks_result, ks)
  end
  return ks_result
end

function ks_less_than(pos1, neg1, pos2, neg2)
  ks_result = Array{Float64,1}()
  for cat in 1:10
    ks = 0
    for score in 0:0.1:10
      p = percent_less_than(score, pos1[cat], neg1[cat])
      q = percent_less_than(score, pos2[cat], neg2[cat])
      ks = max(abs(p - q), ks)
    end
    push!(ks_result, ks)
  end
  return ks_result
end

function seek_alpha(dnm, m, n)
  c(a) = sqrt(-log(a/2) * 1/2)
  cuttoff(a) = c(a) * sqrt((n + m) / (n * m))

  a = 2
  for step_size=2:15
    step = 10.0^(-step_size)
    while (a - step) > 0 && dnm > cuttoff(a - step)
      a -= step
    end
  end

  return a
end

function main()
  DATA = "2_scored_test_data.dat"
  ((test1_pos, test1_neg), (test2_pos, test2_neg)) = Serialization.deserialize(DATA)

  @show map(length, test1_pos)
  @show map(length, test1_neg)
  @show sum(map(length, test1_pos)) + sum(map(length, test1_neg))
  println()

  @show map(length, test2_pos)
  @show map(length, test2_neg)
  @show sum(map(length, test2_pos)) + sum(map(length, test2_neg))
  println()

  ks_greater = ks_greater_than(test1_pos, test1_neg, test2_pos, test2_neg)
  @show ks_greater

  alphas_greater = Array{Float64, 1}()
  for i=1:10
    m = length(test1_pos[i]) + length(test1_neg[i])
    n = length(test2_pos[i]) + length(test2_neg[i])
    alpha = seek_alpha(ks_greater[i], m, n)

    # println("sqrt(-log(a/2)/2) * sqrt(($n+$m) / ($n*$m)) = $(ks_greater[i])    [a should be equal to $alpha")

    push!(alphas_greater, alpha)
  end
  @show alphas_greater

  println()

  ks_lesser = ks_less_than(test1_pos, test1_neg, test2_pos, test2_neg)
  @show ks_lesser

  alphas_lesser = Array{Float64, 1}()
  for i=1:10
    m = length(test1_pos[i]) + length(test1_neg[i])
    n = length(test2_pos[i]) + length(test2_neg[i])

    alpha = seek_alpha(ks_lesser[i], m, n)

    # println("sqrt(-log(a/2)/2) * sqrt(($n+$m) / ($n*$m)) = $(ks_lesser[i])    [a should be equal to $alpha")

    push!(alphas_lesser, alpha)
  end
  @show alphas_lesser


  println()
  println("Writing score details to disk")

  f = open("3_scoring_extra_info.csv", "w")
  confidence_plus_softmax(f, 1, test1_pos, test1_neg)
  confidence_plus_softmax(f, 2, test2_pos, test2_neg)
  close(f)

  println()
  println("Writing plots")

  plots = Array{Plot,1}()
  function render_scatter_plot(set_name, category, pos, neg)
    values = DataFrames.DataFrame(score = [], correct = [], confidence=[])
    for e = pos
      push!(values, Dict(:score => e.score,
                         :correct => true,
                         :confidence => percent_greater_than(e.score, pos, neg)))
    end
    for e = neg
      push!(values, Dict(:score => e.score,
                         :correct => false,
                         :confidence => percent_greater_than(e.score, pos, neg)))
    end

    return plot(values, Guide.title("$set_name - Digit $(category-1)"),
                x=:score, y=:confidence, color=:correct, Geom.point,
                Scale.x_continuous(minvalue=0, maxvalue=10.0),
                Scale.y_continuous(minvalue=0.85, maxvalue=1.0))
  end

  for cat=1:10
    push!(plots, render_scatter_plot("Set 1", cat, test1_pos[cat], test1_neg[cat]))
    push!(plots, render_scatter_plot("Set 2", cat, test2_pos[cat], test2_neg[cat]))
  end

  set_default_plot_size(6inch, 80inch)
  vstack(plots) |> PDF("3_scatterplot.pdf")

  println("all done.")
end

main()
