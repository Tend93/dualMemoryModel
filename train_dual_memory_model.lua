require 'torch'
require 'rnn'
require 'optim'

------ for double state --------------------------------------
local utils = require 'tools.utils'
require 'tools.DataLoader'

local net_utils = require 'tools.net_utils'
-- ====================================================================================================
-- use command line options for model and training configuration
-- I may not be using some of these options in this example
-- ====================================================================================================
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a frame-level encoder-decoder sequence model for video captioning')
cmd:text()
cmd:text('Options')
-- Data
-- Data input and output settings
cmd:option('-input_h5','data/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_h5_local','data/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/data.json','path to the json file containing additional info and vocab')
cmd:option('-topic_file','','topic file')
cmd:option('-label_file', '', 'path to the file for labels')
cmd:option('-add_supervision', 0, 'whether adding extra supervsion info, 1 for true, 0 for false, else for no attention')
cmd:option('-flag','lrtest','path to the json file for vis some results')

-- model params
cmd:option('-hiddensize', 512, 'size of LSTM internal state')


-- optimization
cmd:option('-learningRate',2e-4,'learning rate')
cmd:option('-learning_rate_decay', 5e-5 ,'learning rate decay')
cmd:option('-learning_rate_decay_after',-1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-learning_rate_decay_every', 5, 'every how many epochs thereafter to drop LR by half?')
cmd:option('-dropout',0.8,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-grad_clip',5,'clip gradients at this value, pass 0 to disable')

-- iteration configure
cmd:option('-batchsize', 64,'number of sequences to train on in parallel')
cmd:option('-garbage', 100, 'iter for storage garbage collection, correlating with batchsize, every garbage*batchsize samples')
cmd:option('-max_epochs',500,'number of full passes through the training data')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')

-- Evaluation/Checkpointing
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-checkpoint_dir', 'checkpoints/', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-language_eval',1 , 'do language eval, computer blue and other metrics, 0 = no computer')
cmd:option('-eval_every', 1500 , 'do language eval, computer blue and other metrics, 0 = no computer')

-- SYSTEM SETTING GPU/CPU
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-gpuid',1,'which gpu to use. -1 = use CPU')
cmd:option('-isShuffle', true, 'shuffle the train data for every epoch')
cmd:text()

-- ====================================================================================================
-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- ====================================================================================================
-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        require 'cudnn'
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- =======================================================================
-- load corresponding model
-- =======================================================================
if opt.add_supervision == 0 then
  require 'tools.Seq2Seq_double_state_attention'
elseif opt.add_supervision ==1 then
  require 'tools.Seq2Seq_double_state_attention_supervision'
else
  require 'tools.Seq2Seq_double_state'
end

-- ====================================================================================================
-- -- load data
-- ====================================================================================================
local loadopt = {}
loadopt.h5_file = opt.input_h5
loadopt.h5_file_local = opt.input_h5_local
loadopt.json_file = opt.input_json
loadopt.label_file = opt.label_file
loadopt.topic_file = opt.topic_file
local loader = DataLoader(loadopt)
opt.featsize = loader.feat_width
opt.feat_seq_length = loader.feat_seq_length
opt.topic_size = loader.topic_size
opt.sentence_length = loader.seq_length
-- ====================================================================================================
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- ====================================================================================================
---show some parameters
opt.vocabsize = loader:getVocabSize()
print('vocab size:', opt.vocabsize)
print('learningRate:',opt.learningRate,'sentence_length:',opt.sentence_length)
print('feat_seq_length',opt.feat_seq_length)
print('batchsize:',opt.batchsize)

-- ====================================================================================================
local date_info = os.date('%y%m%d-%H%M')
print('Current time info: ',date_info)
-- some thing for vis --------------------------
--this is used for pastalog to visualization----
------------------------------------------------
opt.modelName = string.format('%s',opt.flag)

-- ====================================================================================================
-- Build the model
-- ====================================================================================================

local protos = {}

local lmopt = {}
lmopt.vocabsize = opt.vocabsize
lmopt.featsize = opt.featsize
lmopt.hiddensize = opt.hiddensize
lmopt.feat_seq_length = opt.feat_seq_length
lmopt.sentence_length = opt.sentence_length
lmopt.topic_size = opt.topic_size
-- for local features
lmopt.att_length = loader.region_num  -- 49
lmopt.att_feat_size = loader.local_width  -- 1024


protos.lm = nn.Seq2Seq(lmopt)
protos.dis = nn.MultiLabelMarginCriterion()
protos.criterion = nn.LanguageModelCriterion()


-- ====================================================================================================
-- run on gpu if possible
-- set all in the model protos to cuda fashion
-- ====================================================================================================
if opt.gpuid >=0 then
   for k,v in pairs(protos) do v:cuda() end
end
-- ====================================================================================================
-- capture all parameters in a single 1-D array
params, grad_params = protos.lm:getParameters()
print('total number of parameters in LM: ', params:nElement())
assert(params:nElement() == grad_params:nElement())


-------------------------------------------------------------------------------------------------------
----- clone of sequence module ------------------------------------------------------------------------

protos.lm:createClones()
-- ====================================================================================================
-- then initilizaton of paramsters
-- ====================================================================================================
-- Uniform Initialization
-- if not (string.len(opt.init_from) > 0) then
--   params:uniform(-0.01, 0.01)
-- end
-- ====================================================================================================
-- evaluation function define
-- ====================================================================================================
function evalLoss(split_index)

   loader:resetIterator(split_index)
   -- opt.alpha_weight:zero()
   -- set evaluation mode
   protos.lm:evaluate()
   local n = 0
   local predictions = {}
   local vocab = loader:getVocab()
   sumError = 0

   while true do
      local data = loader:getBatch{batchsize = opt.batchsize, split = split_index}
      local encInSeq, decInSeq, local_feats = data.images, data.labels, data.local_feats
      n = n + data.images:size(2)  -- batch_size x seq_length x feature width
      if opt.gpuid >= 0 then
         encInSeq = encInSeq:float():cuda()
         decInSeq = decInSeq:float():cuda()
         local_feats = local_feats:float():cuda()
      end
		
      -- forward
      -- local logprobs = protos.lm:forward{encInSeq, decInSeq,local_feats}
      -- local loss = protos.criterion:forward(logprobs, decInSeq)
      --sumError = 0
      
      -- -- forward the model to generate samples for each image
      local opt_sample = {beam_size=5}
      --local opt_sample = {}
      local seq, seqLogprobs = protos.lm:sample({encInSeq,local_feats}, opt_sample)
      local sents = net_utils.decode_sequence(vocab, seq)  -- the last word is the end for fictitious sentence
      --opt.alpha_weight[{{n-opt.batchsize+1,n},{1,alpha:size(2)}}] = alpha
      for k=1,#sents do
         local entry = {image_id = data.infos[k].id, caption = sents[k]}
         table.insert(predictions, entry)
         print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
       
      if n % 50 == 0 then collectgarbage() end
      if data.bounds.wrapped then 
        --debugger.enter()
        break
      end -- the split ran out of data, lets break out
   end
   -- language 
   local lang_stats
   if opt.language_eval == 1 then
      local id = opt.id 
      lang_stats = net_utils.language_eval(predictions, id)
   end
   lang_stats['globalStep'] = loader.epoch - 1 
   --table.insert(opt.metric, lang_stats)
   -- set training mode
   protos.lm:training()

   -- return avg validation loss
   return lang_stats
end

-- ====================================================================================================
-- function for training with optim package
-- ====================================================================================================
function feval(x)
   if x ~= params then
      params:copy(x)
   end

   --  reset gradients
   grad_params:zero()
   -----------------------------------------------------------------------------
   -- Forward pass
   -----------------------------------------------------------------------------
   -- get batch of data  
   local data = loader:getBatchAll{batchsize = opt.batchsize, split = 'train'}
   local encInSeq, decInSeq, local_feats = data.images, data.labels, data.local_feats
   if opt.gpuid >= 0 then
       encInSeq = encInSeq:float():cuda()
       decInSeq = decInSeq:float():cuda()
       local_feats = local_feats:float():cuda()
   end
   local train_loss = {}
   --  forward pass
    if opt.add_supervision == 1 then
      local topics = data.topics
      topics = topics:float():cuda()

      local logprobs, probtopic= protos.lm:forward{encInSeq, decInSeq,local_feats}
      
      train_loss[1] = protos.criterion:forward(logprobs, decInSeq)
      --- probtopic: feat_seq_length* batch_size* topic_size  while topics: batch_size*topic_size
      train_loss[2] = protos.dis:forward(probtopic, topics)
      --  backward pass
      local criOutput = protos.criterion:backward(logprobs, decInSeq)
      local criTopics = protos.dis:backward(probtopic, topics)
      local dlogprobs = protos.lm:backward({encInSeq,decInSeq,local_feats}, {criOutput, criTopics})

    else

      local logprobs= protos.lm:forward{encInSeq, decInSeq,local_feats}
      train_loss[1] = protos.criterion:forward(logprobs, decInSeq)
      train_loss[2] = 0
      --  backward pass
      local criOutput = protos.criterion:backward(logprobs, decInSeq)
      local dlogprobs = protos.lm:backward({encInSeq,decInSeq,local_feats}, criOutput)
    end
   -- clip gradient element-wise (not default)
   if opt.grad_clip > 0 then grad_params:clamp(-opt.grad_clip, opt.grad_clip) end
   
   return train_loss, grad_params
end

-- ==================================================================================
-- Main loop
-- ==================================================================================
-- get training data    TODO: do batching for real data as well.
local ntrain = #loader.shuffle_all  --amount of videos for train
local iterations = opt.max_iters>0 and opt.max_iters or opt.max_epochs * 999999
print('The number of data for a epoch is ', ntrain)
-- store stuff
--local val_loss
local epoch_pre = 1
--local result_history = ''
-- training with optim package

local optim_state = {learningRate = opt.learningRate}
opt.id = 0
--_ = evalLoss('val')

for iter = 1, iterations do

   local epoch = loader.epoch

   -- every now and then or on last iteration
   --if iter % opt.eval_every == 0 then
   if epoch ~= epoch_pre then
      -- evaluate loss on validation data
      local lang_stats
      opt.id = string.format('id%d_dateinfo_%s',epoch,date_info)   
      lang_stats = evalLoss('test')

      local savefile = string.format('%s/%s_%s_%d.t7', opt.checkpoint_dir, date_info, opt.flag, epoch) 
      --local save_alpha = string.format('%s/alpha_%s_%s_%d.t7', opt.checkpoint_dir, date_info, opt.flag, epoch)           
      print('saving checkpoint to ' .. savefile)
      local checkpoint = {}
      checkpoint.opt = opt
      checkpoint.opt.learningRate = optim_state.learningRate
      checkpoint.iter = iter
      checkpoint.epoch = epoch
      print(lang_stats['METEOR'])
      if lang_stats['METEOR']>=0.30 then
         checkpoint.lm = protos.lm:exportModel()
         torch.save(savefile, checkpoint)
         --torch.save(save_alpha, opt.alpha_weight)
         print('have done...')   
      end
      epoch_pre = epoch

      -------------------------------------------------------------------------------------------------
      -- Learning rate decay rules
      -------------------------------------------------------------------------------------------------
      if epoch > opt.learning_rate_decay_after and (epoch-1) % opt.learning_rate_decay_every == 0 then
         local decay_factor = 0.5
         if optim_state.learningRate < 1e-5 then decay_factor = 1 end
         optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
         print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
      end

   end

   local timer = torch.Timer()

   local _, loss = optim.rmsprop(feval, params, optim_state)
   --local _, loss  = optim.adam(feval, params, optim_state)

   local time = timer:time().real
   local train_loss = loss[1] -- the loss is inside a list, pop it
   
   if iter % opt.print_every == 0 then
      print(string.format("%d (epoch %d-%d), train_loss = %6.8f,dis_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", iter, epoch, ntrain, train_loss[1],train_loss[2], grad_params:norm() / params:norm(), time))
   end
   
   if iter % opt.garbage == 0 then collectgarbage() end
   -- if train_loss_old == nil then train_loss_old = train_loss[1] end
   -- if train_loss[1] > train_loss_old * 3 then
   --    print('loss is exploding, aborting.')
   --    break -- halt
   -- end
end

print ('All tasks were done........')

