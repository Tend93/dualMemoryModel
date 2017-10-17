-------------------------------------------------------------------------------------
------[dual memory recurrent model no decoder attention]--
-------------------------------------------------------------------------------------
require 'nn'
require 'torch'
local LSTM_tools = require 'tools.LSTM_double_state'

local layer, parent = torch.class('nn.Seq2Seq', 'nn.Module')

function layer:__init(opt)
  parent.__init(self)
   
  self.continues = false   -- for forward and backward pair judge
  self.featsize = opt.featsize -- the input width of video feature
  self.vocabsize = opt.vocabsize -- the size of vocabulary
  self.hiddensize = opt.hiddensize

  self.sentence_length = opt.sentence_length
  self.feat_seq_length = opt.feat_seq_length
  self.word_encoding_size = 512
  self.topic_size = opt.topic_size
  self.reason_weight = 0
  self.hiddensize_dec = 1024
  print('reason weight is ',self.reason_weight)
  -- build up models
  print('build double state model....')
  print(opt)

  local opt_enc = {}
  opt_enc.input_size = opt.featsize
  opt_enc.rnn_size = opt.hiddensize

  opt_enc.att_type = 3   -- set attention type 1:local_prev_h ,2:prev_h, 3:content , default is 1
  opt_enc.att_feat_size = opt.att_feat_size
  opt_enc.att_length = opt.att_length
  opt_enc.topic_size = opt.topic_size
  opt_enc.att_hid_size = 100

  local opt_dec = {}
  opt_dec.input_size = self.word_encoding_size  -- word_encoding_size
  opt_dec.rnn_size = self.hiddensize_dec
  opt_dec.output_size = opt.vocabsize + 1
 
  opt_dec.att_feat_size = opt.featsize
  opt_dec.att_hid_size = 100
  opt_dec.att_length = opt.feat_seq_length

  self.encoder = LSTM_tools.lstm_soft_att_double_state(opt_enc) -- feature encoder
  self.decoder = LSTM_tools.lstm_language(opt_dec) -- sentence decoder
  self.embed = nn.LookupTableMaskZero(opt.vocabsize + 1 , self.word_encoding_size)
  self.connectTable_c = nn.JoinTable(2)
  self.connectTable_h = nn.JoinTable(2)
  print('done....')
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the sequence LSTM model')
  self.encoder_clones = {self.encoder}
  self.decoder_clones = {self.decoder}
  self.embed_clones = {self.embed}
  -- encoder part
  for t=2,self.feat_seq_length do
    self.encoder_clones[t] = self.encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
 
  --  decoder part
  for t=2,self.sentence_length + 1 do
    self.decoder_clones[t] = self.decoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.embed_clones[t] = self.embed:clone('weight', 'gradWeight')
  end

end

function layer:getModulesList()
  return {self.encoder, self.decoder, self.embed}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.encoder:parameters()
  local p2,g2 = self.decoder:parameters()
  local p3,g3 = self.embed:parameters()


  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  for k,v in pairs(p3) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  for k,v in pairs(g3) do table.insert(grad_params, v) end

  return params, grad_params
end

function layer:training()
  if self.encoder_clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.encoder_clones) do v:training() end
  for k,v in pairs(self.decoder_clones) do v:training() end
  for k,v in pairs(self.embed_clones) do v:training() end
  
end

function layer:evaluate()
  if self.encoder_clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.encoder_clones) do v:evaluate() end
  for k,v in pairs(self.decoder_clones) do v:evaluate() end
  for k,v in pairs(self.embed_clones) do v:evaluate() end
  
end


function layer:updateOutput(input)   -- input = {global feature, word feature, local feature}
  local context = input[1]   --10*64*1000  sequence length * batch size * feature size
  local words = input[2]     --16*64       sentence length * batch size
  local local_feats = input[3]

  if self.encoder_clones == nil then self:createClones() end -- lazily create clones on first forward pass
  --debugger.enter()
  assert(context:size(1) == self.feat_seq_length , 'context length wrong')
  assert(words:size(1) == self.sentence_length , 'sentence_length length wrong')

  local batch_size = context:size(2)

  self.outputs = torch.CudaTensor(self.sentence_length+1, batch_size, self.vocabsize+1):zero()

  local enc_ct={[0] = torch.CudaTensor(batch_size, self.hiddensize):zero()}
  local enc_ht={[0] = torch.CudaTensor(batch_size, self.hiddensize):zero()}
  local local_enc_ct={[0] = torch.CudaTensor(batch_size, self.hiddensize):zero()}
  local local_enc_ht={[0] = torch.CudaTensor(batch_size, self.hiddensize):zero()}
  local dec_ct={}
  local dec_ht={}

  self.enc_inputs = {}
  self.dec_inputs = {}
  self.embed_inputs = {}

  for t=1, self.feat_seq_length do
      self.enc_inputs[t] = {context[t],local_feats[t],enc_ct[t-1], enc_ht[t-1],local_enc_ct[t-1], local_enc_ht[t-1]}
      local enc_out = self.encoder_clones[t]:forward(self.enc_inputs[t])
      enc_ct[t], enc_ht[t],local_enc_ct[t], local_enc_ht[t] = unpack(enc_out)
  end

  -- for  encoder and decoder connection ------
  self.connect_input_c = {enc_ct[self.feat_seq_length], local_enc_ct[self.feat_seq_length]}
  self.connect_input_h = {enc_ht[self.feat_seq_length], local_enc_ht[self.feat_seq_length]}
  dec_ct[0] = self.connectTable_c:forward(self.connect_input_c)
  dec_ht[0] = self.connectTable_h:forward(self.connect_input_h)

  self.dec_max = 0    -- index for max decoder steps
  for t=1, self.sentence_length+1 do
      local embed_out
      if t == 1 then   -- init prev_c and prev_h using encoder output
        local it = torch.LongTensor(batch_size):fill(self.vocabsize+1)
        self.embed_inputs[t] = it
        embed_out = self.embed_clones[t]:forward(it) -- NxK sized input (token embedding vectors)
      else
        -- feed in the rest of the sequence...
        local it = words[t-1]:clone()
        self.embed_inputs[t] = it
        embed_out = self.embed_clones[t]:forward(it)
      end

      self.dec_inputs[t] = {embed_out, dec_ct[t-1], dec_ht[t-1] }
      local dec_out = self.decoder_clones[t]:forward(self.dec_inputs[t])
      dec_ct[t], dec_ht[t], self.outputs[t] = unpack(dec_out)
      self.dec_max = t
  end
  --assert(self.dec_max == self.sentence_length+1 ,' wrong dec_max value ....')
  self.continues = true
  return self.outputs
end

function layer:updateGradInput(input, gradOutput)  
  assert(self.continues == true, 'please run forward operation before backward')
  self.continues = false
  local gradOutput_dec = gradOutput
  local batch_size = gradOutput_dec:size(2)
  local denc_ct={}
  local denc_ht={}
  local local_denc_ct={}
  local local_denc_ht={}
  local ddec_ct={[self.dec_max] = torch.CudaTensor(batch_size, self.hiddensize_dec):zero()}
  local ddec_ht={[self.dec_max] = torch.CudaTensor(batch_size, self.hiddensize_dec):zero()}
  --local alpha_mask = torch.CudaTensor(batch_size,49):zero()
  self.gradInput = 0
  for t=self.dec_max,1,-1 do
      local ddec_out = {ddec_ct[t], ddec_ht[t], gradOutput_dec[t]}
      local dembed
      dembed,ddec_ct[t-1], ddec_ht[t-1] = unpack(self.decoder_clones[t]:backward(self.dec_inputs[t], ddec_out))
      local it = self.embed_inputs[t]
      self.embed_clones[t]:backward(it, dembed)
  end
  
  -- for encoder and decoder connection
  denc_ct[self.feat_seq_length], local_denc_ct[self.feat_seq_length] = unpack(self.connectTable_c:backward(self.connect_input_c,ddec_ct[0]))
  denc_ht[self.feat_seq_length], local_denc_ht[self.feat_seq_length] = unpack(self.connectTable_h:backward(self.connect_input_h,ddec_ht[0]))
  for t=self.feat_seq_length,1,-1 do
    local denc_out = {denc_ct[t], denc_ht[t],local_denc_ct[t], local_denc_ht[t]}
    _,_,denc_ct[t-1], denc_ht[t-1],local_denc_ct[t-1], local_denc_ht[t-1] = unpack(self.encoder_clones[t]:backward(self.enc_inputs[t], denc_out))
  end
  return self.gradInput
end 

-- clear outputs and some intermediate results for model saving 
function layer:setThinModel()
  print('we don\'t use the module here!')
end

function layer:exportModel()
  local encoder = self.encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
  local decoder = self.decoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
  local embeding = self.embed:clone('weight','gradWeight')
  encoder:clearState()
  decoder:clearState()
  embeding:clearState()
  return {encoder, decoder, embeding}
end

function layer:importModel(model)
  self.encoder = model[1]
  self.decoder = model[2]
  self.embed = model[3]
end

function layer:sample(input, opt)

  local context = input[1]
  local local_feats = input[2]
  local batch_size = context:size(2)

  local enc_ct = torch.CudaTensor(batch_size, self.hiddensize):zero()
  local enc_ht = torch.CudaTensor(batch_size, self.hiddensize):zero()
  local local_enc_ct = torch.CudaTensor(batch_size, self.hiddensize):zero()
  local local_enc_ht = torch.CudaTensor(batch_size, self.hiddensize):zero()
  local dec_ct, dec_ht
  for t=1, self.feat_seq_length do
    local enc_out = self.encoder:forward{context[t],local_feats[t],enc_ct,enc_ht,local_enc_ct, local_enc_ht}
    enc_ct, enc_ht,local_enc_ct, local_enc_ht= unpack(enc_out)
  end
  -- encoder and decoder connection
  dec_ct = self.connectTable_c:forward{enc_ct, local_enc_ct}
  dec_ht = self.connectTable_h:forward{enc_ht, local_enc_ht}

  if opt.beam_size ~= nil then return self:beam_search({dec_ct,dec_ht}, {beam_size = opt.beam_size}) end
  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.sentence_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.sentence_length, batch_size):zero()
  local logprobs 
  for t=1,self.sentence_length+1 do
      local embed_out, it, sampleLogprobs

      if t==1 then
        it = torch.LongTensor(batch_size):fill(self.vocabsize+1)
        embed_out = self.embed:forward(it)
      else
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
        embed_out = self.embed:forward(it)
      end
      if t >= 2 then 
        seq[t-1] = it -- record the samples
        seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
      end
      local dec_input = {embed_out,dec_ct, dec_ht}
      dec_ct, dec_ht, logprobs = unpack(self.decoder:forward(dec_input))
  end

  return seq, seqLogprobs
end

function layer:beam_search(input, opt)
  print('doing beam serch with beam size ', opt.beam_size)
  local batch_size = input[1]:size(1)
  local beam_size = opt.beam_size

  local function compare(a,b) return a.p > b.p end -- used downstream

  local seq = torch.LongTensor(self.sentence_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.sentence_length, batch_size):zero()

  for k=1, batch_size do

    local state_c = input[1]:narrow(1,k,1):expand(beam_size, self.hiddensize_dec)
    local state_h = input[2]:narrow(1,k,1):expand(beam_size, self.hiddensize_dec)
    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.sentence_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.sentence_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    --local att_seq_temp = att_seq:narrow(1,k,1):expand(beam_size, self.feat_seq_length, feat_dim):contiguous() 
    for t=1,self.sentence_length+1 do
      local embed_out, it, sampleLogprobs
      local new_state_c, new_state_h
      if t==1 then
        it = torch.LongTensor(beam_size):fill(self.vocabsize+1)
        embed_out = self.embed:forward(it)
      else
        local logprobsf = logprobs:float()
        ys, ix = torch.sort(logprobsf,2, true)
        local candidates = {}
        local cols = beam_size
        local rows = beam_size
        if t==2 then rows = 1 end  -- when t=2, we only see a beam sample for all samples are the same. pnly a row
        for c=1, cols do 
          for q=1, rows do
            local local_logprob = ys[{q,c}]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates,{c=ix[{q,c}], q=q, p=candidate_logprob, r=local_logprob})
          end
        end
        table.sort(candidates, compare)

        -- construct new beams
        new_state_c = state_c:clone()
        new_state_h = state_h:clone()

        local beam_seq_prev, beam_seq_logprobs_prev
        if t>2 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()   -- when t=3, we begion to record words and logprobs
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        end
        for vix=1, beam_size do
          local v = candidates[vix]
          if t > 2 then
            beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end

          -- arrange states 
          new_state_c[vix] = state_c[v.q]
          new_state_h[vix] = state_h[v.q]

          beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam
          if v.c == self.vocabsize+1 or t == self.sentence_length+1 then
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end

        it = beam_seq[t-1]
        embed_out = self.embed:forward(it)
      end
      if new_state_c ~= nil then state_c, state_h = new_state_c:clone(), new_state_h:clone() end
      ----------- we get the best beam_size words -------
      local dec_inputs = {embed_out, state_c, state_h}
      local dec_out = self.decoder:forward(dec_inputs)
      state_c, state_h, logprobs = unpack(dec_out)
    end

    table.sort(done_beams, compare)
    seq[{{},k}] = done_beams[1].seq
    seqLogprobs[{{},k}] = done_beams[1].logps
  end
  return seq, seqLogprobs
end


-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end


function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3) -- L sequences
  local D = seq:size(1)                                       -- N batchsize
  assert(D == L-1, 'input Tensor should be 1 larger in time') -- Mp1 vocabsize+1
                                                              -- D label sequences = L-1
  local loss = 0
  local n = 0
  local num_correct = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=1,L do -- iterate over sequence time (forward for the <BoS>)

      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0        -- the final prob output doesn't calculate loss
      else
        target_index = seq[{t,b}] -- t is correct, since at t=1 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
    
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1            -- regard this as an end token
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
