# modified from https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl

using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using MLDatasets
using Serialization

@with_kw mutable struct Args3
  η::Float64 = 3e-4       # learning rate
  segment = 0             # 0-9
  batchsize::Int = 1024   # batch size
  epochs::Int = 10        # number of epochs
  throttle::Int = 1		    # Throttle timeout
end

function load_shuffled_dataset()
  # load and combined the train and test data
  combined_x = zeros(Float32, 28, 28, 70_000)
  combined_x[:,:,1:60_000] = MLDatasets.MNIST.traindata(Float32)[1]
  combined_x[:,:,60_001:70_000] = MLDatasets.MNIST.testdata(Float32)[1]

  combined_y = zeros(Int64, 70_000)
  combined_y[1:60_000] = MLDatasets.MNIST.traindata(Float32)[2]
  combined_y[60_001:70_000] = MLDatasets.MNIST.testdata(Float32)[2]

  # apply the ordering/removal dictated by the static shuffle
  static_shuffle = Serialization.deserialize("1_static_shuffle.dat")
  shuffled_x = combined_x[:, :, static_shuffle[:]]
  shuffled_y = combined_y[static_shuffle[:]]

  return (shuffled_x, shuffled_y)
end

# cache loaded dataset
const (SHUFFLED_X, SHUFFLED_Y) = load_shuffled_dataset()

function get_segment(segment, batchsize)
  # segmentation for cross validation
  # training set is 1:total_size in SHUFFLED_X and SHUFFLED_Y
  # testing sets(!) are at -20_000:end in SHUFFLED_X and SHUFFLED_Y

  total_size = length(SHUFFLED_Y) - 20_000
  segment_size = Int64(floor(total_size / 10))
  segment_start = segment * segment_size
  segment_end = segment_start + segment_size

  @show total_size, segment_size
  train_x, train_y = zeros(Float32, (28, 28, 9 * segment_size)), zeros(Float32, 9 * segment_size)
  i = 1
  offset = 0
  for i=1:(9 * segment_size)
    if i == segment_start
      offset = segment_size
    end

    train_x[:,:,i] = SHUFFLED_X[:,:,i + offset]
    train_y[i] = SHUFFLED_Y[i + offset]
  end

  validate_x, validate_y = zeros(Float32, (28, 28, segment_size)), zeros(Float32, segment_size)
  for i=1:segment_size
    validate_x[:,:,i] = SHUFFLED_X[:,:,i+segment_start]
    validate_y[i] = SHUFFLED_Y[i+segment_start]
  end

  # Reshape Data for flatten the each image into linear array
  train_x, validate_x = Flux.flatten(train_x), Flux.flatten(validate_x)

  # One-hot-encode the labels
  train_y, validate_y = onehotbatch(train_y, 0:9), onehotbatch(validate_y, 0:9)

  # Batching
  train_data = DataLoader(train_x, train_y, batchsize=batchsize, shuffle=true)
  validate_data = DataLoader(validate_x, validate_y, batchsize=batchsize)

  return train_data, validate_data
end

function get_test_data(batchsize)
  total_length = length(SHUFFLED_Y)
  test1_offset = total_length - 20_000 + 1
  test2_offset = total_length - 10_000 + 1

  test1_x = SHUFFLED_X[:,:,test1_offset:(test2_offset - 1)]
  test1_y = SHUFFLED_Y[test1_offset:(test2_offset - 1)]

  test2_x = SHUFFLED_X[:,:,test2_offset:total_length]
  test2_y = SHUFFLED_Y[test2_offset:total_length]

  # Reshape Data for flatten the each image into linear array
  test1_x = Flux.flatten(test1_x)
  test2_x = Flux.flatten(test2_x)

  # Batching
  test1 = DataLoader(test1_x, test1_y, batchsize=batchsize)
  test2 = DataLoader(test2_x, test2_y, batchsize=batchsize)

  return ((test1_x, test1_y), (test2_x, test2_y))
end

function build_model(; imgsize=(28,28,1), nclasses=10)
  return Chain(
 	  Dense(prod(imgsize), 32, relu),
    Dense(32, nclasses),
    softmax)
end

function loss_all(dataloader, model)
  l = 0f0
  for (x,y) in dataloader
    l += sum((model(x) .- y) .^ 2)
  end
  l/length(dataloader)
end

function accuracy(data_loader, model)
  acc = 0
  for (x,y) in data_loader
    acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
  end
  acc/length(data_loader)
end

function train(args)
  train_data, test_data = get_segment(args.segment, args.batchsize)

  # Construct model
  m = build_model()
  loss(x,y) = logitcrossentropy(m(x), y)

  ## Train model
  evalcb = throttle(() -> @show(loss_all(train_data, m)), args.throttle)
  opt = ADAM(args.η)

  @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)
  @show accuracy(train_data, m)
  @show accuracy(test_data, m)

  m
end

function train_models(; kws...)
  # Initializing Model parameters
  args = Args3(; kws...)

  models = []
  for i=0:9
    args.segment = i
    push!(models, train(args))
  end

  println("Done training!")

  return models
end


struct RunResult
  output :: Array{Float32, 1}
  max_output_i :: Int64
  score :: Float32
  correct_category :: Int64
end

function score_data(models, x, y)
  true_pos  = [[], [], [], [], [], [], [], [], [], []]
  false_pos = [[], [], [], [], [], [], [], [], [], []]

  res = zeros(10, length(y))
  for m in models
    res .+= m(x)
  end

  for i=1:size(res, 2)
    (score, idx) = findmax(res[:,i])

    r = RunResult(copy(res[:,i]), idx-1, score, y[i])

    if r.max_output_i == r.correct_category
      push!(true_pos[idx], r)
    else
      push!(false_pos[idx], r)
    end
  end

  return true_pos, false_pos
end

function score_test_data(models; kws...)
  args = Args3(; kws...)

  ((test1_x, test1_y), (test2_x, test2_y)) = get_test_data(args.batchsize)

  t1 = score_data(models, test1_x, test1_y)
  t2 = score_data(models, test2_x, test2_y)

  return (t1, t2)
end

MODEL_FILE_NAME = "2_model.dat"
DATA_FILE_NAME = "2_scored_test_data.dat"

function main()
  if isfile(MODEL_FILE_NAME)
    println("Found cached model, loading.")
    models = Serialization.deserialize(MODEL_FILE_NAME)

    if !isfile(DATA_FILE_NAME)
      println("Generating model output.")
      scored = score_test_data(models)
      Serialization.serialize(DATA_FILE_NAME, scored)
      println("All done.")
    else
      println("Found cached model output, all done.")
    end
  else
    println("Training the model.")
    models = train_models()
    Serialization.serialize(MODEL_FILE_NAME, models)

    # We've just trained a new model, so cached classification is invalid
    if isfile(DATA_FILE_NAME)
      rm(DATA_FILE_NAME)
    end

    println("Generating model output.")
    scored = score_test_data(models)
    Serialization.serialize(DATA_FILE_NAME, scored)
    println("All done.")
  end
end

main()
