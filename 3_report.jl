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
  pos_len = length(filter(e -> e.score >= x, pos)) + 1e-30 # prevent divide by 0
  neg_len = length(filter(e -> e.score >= x, neg))
  return pos_len / (pos_len + neg_len)
end

function percent_less_than(x, pos, neg)
  pos_len = length(filter(e -> e.score <= x, pos))
  neg_len = length(filter(e -> e.score <= x, neg)) + 1e-30 # prevent divide by 0
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

  println("Test set statistics -- test1:")
  @show map(length, test1_pos)
  @show map(length, test1_neg)
  @show sum(map(length, test1_pos)) + sum(map(length, test1_neg))
  println()

  println("Test set statistics -- test2:")
  @show map(length, test2_pos)
  @show map(length, test2_neg)
  @show sum(map(length, test2_pos)) + sum(map(length, test2_neg))
  println()

  println("Writing KS metrics to disk")
  println()

  f = open("3_ks_stats.csv", "w")
  println(f, "greater_than, category, ks, alpha")
  begin
    ks_greater = ks_greater_than(test1_pos, test1_neg, test2_pos, test2_neg)
    @show ks_greater

    alphas_greater = Array{Float64, 1}()
    for i=1:10
      m = length(test1_pos[i]) + length(test1_neg[i])
      n = length(test2_pos[i]) + length(test2_neg[i])
      alpha = seek_alpha(ks_greater[i], m, n)

      #println("sqrt(-log(a/2)/2) * sqrt(($n+$m) / ($n*$m)) = $(ks_greater[i])    [a should be equal to $alpha")

      push!(alphas_greater, alpha)
    end
    @show alphas_greater

    for i=1:10
      println(f, "1, $i, $(ks_greater[i]), $(alphas_greater[i])")
    end
  end

  begin
    ks_lesser = ks_less_than(test1_pos, test1_neg, test2_pos, test2_neg)
    @show ks_lesser

    alphas_lesser = Array{Float64, 1}()
    for i=1:10
      m = length(test1_pos[i]) + length(test1_neg[i])
      n = length(test2_pos[i]) + length(test2_neg[i])

      alpha = seek_alpha(ks_lesser[i], m, n)

      #println("sqrt(-log(a/2)/2) * sqrt(($n+$m) / ($n*$m)) = $(ks_lesser[i])    [a should be equal to $alpha")

      push!(alphas_lesser, alpha)
    end
    @show alphas_lesser

    for i=1:10
      println(f, "0, $i, $(ks_lesser[i]), $(alphas_lesser[i])")
    end
  end

  close(f)


  println()
  println("Writing score details to disk")

  f = open("3_scoring_full_data.csv", "w")
  confidence_plus_softmax(f, 1, test1_pos, test1_neg)
  confidence_plus_softmax(f, 2, test2_pos, test2_neg)
  close(f)

  println()
  println("Writing plots")

  plots = Array{Plot,1}()
  function render_scatter_plot(title, f, category, pos1, neg1, pos2, neg2)
    function make_layers(pos, neg, c1, c2)
      values = DataFrames.DataFrame(Score = [], Correct = [], Confidence=[])
      for e = pos
        push!(values, Dict(:Score => e.score,
                           :Correct => true,
                           :Confidence => f(e.score, pos, neg)))
      end
      l1 = layer(values, x=:Score, y=:Confidence, Geom.point, Theme(default_color=c1, point_size = 0.5 * mm, discrete_highlight_color=c->nothing))

      values = DataFrames.DataFrame(Score = [], Correct = [], Confidence=[])
      for e = neg
        push!(values, Dict(:Score => e.score,
                           :Correct => false,
                           :Confidence => f(e.score, pos, neg)))
      end
      l2 = layer(values, x=:Score, y=:Confidence, Geom.point, Theme(default_color=c2, point_size = 0.5 * mm, discrete_highlight_color=c->nothing))
      return (l1, l2)
    end

    l11, l12 = make_layers(pos1, neg1, "blue", "red")
    l21, l22 = make_layers(pos2, neg2, "green", "orange")

    return plot(l22, l12, l21, l11, Guide.manual_color_key("Correctness",
                                                           ["true pos 1", "false pos 1", "true pos 2", "false pos 2"],
                                                           ["blue", "red", "green", "orange"]),
                Guide.title(title),
                Scale.x_continuous(minvalue=0, maxvalue=10.0),
                Scale.y_continuous(minvalue=0.86, maxvalue=1.0))
  end

  for cat=1:10
    push!(plots, render_scatter_plot("Digit $(cat - 1) - Greater than cumulative", percent_greater_than, cat, test1_pos[cat], test1_neg[cat], test2_pos[cat], test2_neg[cat]))
    push!(plots, render_scatter_plot("Digit $(cat - 1) - Less than cumulative", percent_less_than, cat, test1_pos[cat], test1_neg[cat], test2_pos[cat], test2_neg[cat]))
  end

  set_default_plot_size(12inch, 160inch)
  vstack(plots) |> PDF("3_scatterplot.pdf")

  println("all done.")
end

main()
