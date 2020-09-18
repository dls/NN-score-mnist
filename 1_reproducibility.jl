# segment the datasets in a reproducible way

using Serialization
using MLDatasets
using Random

FILE_NAME = "1_static_shuffle.dat"

if isfile(FILE_NAME)
  println("Found exiting shuffle index, standing down.")
else
  static_shuffle = Array{Int64,1}()

  if length(ARGS) == 0
    println("USAGE: julia 1_reproducibility.jl (MIX|KEEP|TRIM) [prob] [cats...]")
    exit(0)
  end

  if ARGS[1] == "KEEP"
    println("keeping mnist official test set separate.")
    static_shuffle = vcat(shuffle(Array{Int64,1}(1:60_000)), Array{Int64,1}(60_001:70_000))
  else
    if ARGS[1] == "TRIM"
      trim_p = parse(Float64, ARGS[2])
      to_drop = map(e -> parse(Int64,e), ARGS[3:end])

      println("Trimming out $(Int64(round(trim_p * 100)))% of these categories: $to_drop, keeping mnist official test set separate.")

      ys = MLDatasets.MNIST.traindata(Float32)[2]
      for i=1:50_000
        if in(ys[i], to_drop)
          if rand() > (1.0 - trim_p)
            push!(static_shuffle, i)
          end
        else
          push!(static_shuffle, i)
        end
      end
      static_shuffle = vcat(shuffle(static_shuffle), Array{Int64,1}(50_001:70_000))
    else
      if ARGS[1] == "MIX"
        println("mixing the mnist official test in for potential training.")
        static_shuffle = shuffle!(Array{Int64,1}(1:70_000))
      else
        println("Unrecognized command. See README.md")
      end
    end
  end

  Serialization.serialize(FILE_NAME, static_shuffle)

  println("Shuffle index built")
end
