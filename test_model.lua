require 'torch'
require 'rnn'
require 'optim'
require 'nngraph'

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
-- model params

-- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '/home/yzw/github/dualMemoryModel/checkpoints/170929-1513_double_state_attention_just_test_2.t7', 'initialize network parameters from checkpoint at this path')

-- Evaluation/Checkpointing
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-checkpoint_dir', 'checkpoints/', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-language_eval',1 , 'do language eval, computer blue and other metrics, 0 = no computer')

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
---------- load model -----------------------------
print('loading model from:', opt.init_from)
checkpoint = torch.load(opt.init_from)
opt = checkpoint.opt
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
---show some parameters
opt.vocabsize = loader:getVocabSize()
print('vocab size:', opt.vocabsize)
print('learningRate:',opt.learningRate,'sentence_length:',opt.sentence_length)
print('feat_seq_length',loader.feat_seq_length)
print('batchsize:',opt.batchsize)

-- ====================================================================================================
local date_info = os.date('%y%m%d-%H%M')
print('Current time info: ',date_info)
-- some thing for vis --------------------------
--this is used for pastalog to visualization----
------------------------------------------------
opt.modelName = string.format('%s',opt.flag)
opt.vis_info = string.format('h-%s', opt.hiddensize)
opt.metric = {}
opt.trainloss = {}
opt.valloss = {}

-- ====================================================================================================
-- Build the model
-- ====================================================================================================

local protos = {}
-- check if need to load from previous experiment

      

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
protos.lm:importModel(checkpoint.lm)


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

function evalLoss(split_index)

   loader:resetIterator(split_index)
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
      
      -- -- forward the model to generate samples for each image
      local opt_sample = {beam_size=5}
      local seq, seqLogprobs,alpha = protos.lm:sample({encInSeq,local_feats}, opt_sample)
      local sents = net_utils.decode_sequence(vocab, seq)  -- the last word is the end for fictitious sentence

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
      local id = 'test1' 
      lang_stats = net_utils.language_eval(predictions, id)
   end

   -- set training mode
   protos.lm:training()

   -- return avg validation loss
   return lang_stats
end


local lang_stats
lang_stats = evalLoss('test')


print ('All tasks were done........')

