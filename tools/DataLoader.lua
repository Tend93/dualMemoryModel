------ new write 2017-09-05 Taehi ----------------------

require 'hdf5'
local utils = require 'tools.utils'
local DataLoader = torch.class('DataLoader')
local npy4th = require 'npy4th'
function DataLoader:__init(opt)
  
  self.isShuffle = true
  self.epoch = 1
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)              -- for word and vocabulary
  self.ix_to_word = self.info.ix_to_word
  self.vocab_size = utils.count_keys(self.ix_to_word)
  self.list = self.info.img2sent_list                     -- video VS sentence index
  self.list_end_ix = self.info.label_end_ix               -- setence list end for each video
  print('vocab size is ' .. self.vocab_size)
  

  -------- load context features  ---------------
  -- open the hdf5 file
  print('DataLoader loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  
  -- extract image size from dataset
  local images_size = self.h5_file:read('/images'):dataspaceSize()
  assert(#images_size == 3, '/images should be a 3D tensor')
  -- assert(images_size[3] == images_size[4], 'width and height must match')
  self.num_videos = images_size[1]
  self.feat_seq_length= images_size[2]
  self.feat_width = images_size[3]
  print(string.format('read %d videos of size %dx%d', self.num_videos,
            self.feat_seq_length, self.feat_width))

  -------- load local features ---------------
  -- open the hdf5 file
  print('DataLoader loading h5 file: ', opt.h5_file_local)
  self.h5_file_local = hdf5.open(opt.h5_file_local, 'r')

  local images_size_local = self.h5_file_local:read('/images'):dataspaceSize()   -- 64*10*1024*49
  assert(#images_size_local == 4, '/local images should be a 4D tensor')
 
  self.local_seq_length = images_size_local[2]   -- video sequence length  1970*10*1024*49
  self.local_width = images_size_local[3]		 -- local feature dim
  self.region_num = images_size_local[4]       -- local feature map regions
  print(string.format('read local features of size %dx%dx%d', self.local_seq_length,
            self.local_width, self.region_num))


  -- load in the sequence data
  -- load labels from npy file
  ----------- load labels ----------------------------------
  self.labels = npy4th.loadnpy(opt.label_file)

  local seq_size = self.labels:size()
  self.seq_length = seq_size[2]
  print('max sequence length in data is ' .. self.seq_length)

  --local top_k_topic = 5
  print('locading topic file'.. opt.topic_file)
  --print('get top k number '..top_k_topic)
  print('get word word_distribution....')
  --self.topics_raw = npy4th.loadnpy(opt.topic_file)
  --self.topics_raw = self.labels
  --self.topic_size = self.topics_raw:size(2)
  self.topic_size = self.vocab_size
  self.topics_pre = torch.LongTensor(self.num_videos, self.topic_size)
  self.topics = torch.LongTensor(self.num_videos, self.topic_size)
  -- from word distribution ...
  print('preparing word distribution')

  for i=1, seq_size[1] do
    local vd_ix = self.list[tostring(i)]
    for j=1, self.seq_length do
      local word_ix = self.labels[i][j]
      if word_ix ~= 0 then
     	 self.topics_pre[vd_ix][word_ix]=1
      end
    end
  end

  for i=1, self.num_videos do
    local tp_ix = 1
    for j=1, self.topic_size do
        local tp = self.topics_pre[i][j]
        if tp == 1 then
          self.topics[i][tp_ix] = j
          tp_ix = tp_ix+1
        end
    end
  end 

  print('word distribution samples')
  print(self.topics[1])

  self.split_ix = {}
  self.iterators = {}
  self.shuffle_all = {}
  self.iterator_all = 1
  for i,img in pairs(self.info.images) do
    local split = img.split
    if not self.split_ix[split] then
      -- initialize new split
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.split_ix[split], i)
  end

  -- for new training table shuffle  6513 for MSRVTT
  for i=1, self.list_end_ix[1200] do
    table.insert(self.shuffle_all, i)
  end
  utils.shuffle_table(self.shuffle_all)
  -- check shuffle_all table with list_end_ix and list -----
  print('train table label index is',self.list_end_ix[1200])
  assert(self.list[tostring(self.list_end_ix[1200])] == 1200, 'load train sentences index wrong')
  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d images to split %s', #v, k))
  end

end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:getSeqLength()
  return self.seq_length
end

function DataLoader:getBatchAll(opt)
  local split = opt.split  -- lets require that user passes this in, for safety
  local batchsize = opt.batchsize -- how many images get returned at one time (to go through CNN)

  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.FloatTensor(batchsize , self.feat_seq_length, self.feat_width):zero()
  local local_batch_raw = torch.FloatTensor(batchsize, self.local_seq_length, self.local_width, self.region_num):zero()
  local label_batch = torch.LongTensor(batchsize , self.seq_length):zero()
  local topic_batch = torch.LongTensor(batchsize, self.topic_size):zero()
  local wrapped = false
  local infos = {}
  local shuffle_all = self.shuffle_all

  local max_index = #shuffle_all
  
  for i=1, batchsize do

      local ri = self.iterator_all -- the index of current iteration
      ri_next = ri + 1
      if ri_next > max_index then
        ri_next = 1
        wrapped = true
        -- shuffle train split
        utils.shuffle_table(self.shuffle_all)
        print(split .. ' data is shuffled...')
        self.epoch = self.epoch + 1
      end -- wrap back around
      self.iterator_all = ri_next
      --debugger.enter()
      local ix = shuffle_all[ri]

      local ix_img = self.list[tostring(ix)] 
      if ix_img>1200 then print('get wrong training image index...') end
	  --debugger.enter()
      local img = self.h5_file:read('/images'):partial({ix_img,ix_img},
                            {1,self.feat_seq_length},{1, self.feat_width})
      img_batch_raw[i] = img
      

	    local_batch_raw[i] = self.h5_file_local:read('/images'):partial({ix_img,ix_img},
                            {1, self.local_seq_length},{1, self.local_width},{1,self.region_num})
      						
      label_batch[i] = self.labels[{ix,{1,self.seq_length}}]
      topic_batch[i] = self.topics[{ix_img,{1,self.topic_size}}]
      
       -- and record associated info as well
      local info_struct = {}
      info_struct.id = self.info.images[ix_img].id
      table.insert(infos, info_struct)
  end
  local data = {}
  data.local_feats = local_batch_raw:transpose(1,2):transpose(3,4):contiguous()
  data.topics = topic_batch:contiguous()
  data.images = img_batch_raw:transpose(1,2):contiguous()  --note: sequencer input data shuld be seq_len*batchsize*feat_len
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.bounds = {it_pos_now = self.iterator_all, it_max = max_index, wrapped = wrapped}
  data.infos = infos
  
  return data
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - X (N,3,H,W) containing the images
  - y (L,M) containing the captions as columns (which is better for contiguous memory during training)
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
  local split = opt.split -- lets require that user passes this in, for safety
  local batchsize = opt.batchsize -- how many images get returned at one time (to go through CNN)

  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  -- pick an index of the datapoint to load next
  local img_batch_raw = torch.FloatTensor(batchsize , self.feat_seq_length, self.feat_width):zero()
  local local_batch_raw = torch.FloatTensor(batchsize, self.local_seq_length, self.local_width, self.region_num):zero()
  local label_batch = torch.LongTensor(batchsize, self.seq_length):zero()
  local topic_batch = torch.LongTensor(batchsize, self.topic_size):zero()

  local max_index = #split_ix
  local wrapped = false
  local infos = {}
  for i=1,batchsize do

    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then
      ri_next = 1
      wrapped = true
    end -- wrap back around
    self.iterators[split] = ri_next
    ix = split_ix[ri]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

    -- fetch the image from h5
    local img = self.h5_file:read('/images'):partial({ix,ix},
                            {1,self.feat_seq_length},{1, self.feat_width})
    img_batch_raw[i] = img
	
	
    local_batch_raw[i] = self.h5_file_local:read('/images'):partial({ix,ix},
                          {1, self.local_seq_length},{1, self.local_width},{1,self.region_num})
	
    local info_struct = {}
    info_struct.id = self.info.images[ix].id
    table.insert(infos, info_struct)
  end

  local data = {}
  data.local_feats = local_batch_raw:transpose(1,2):transpose(3,4):contiguous()
  data.images = img_batch_raw:transpose(1,2):contiguous()
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.topics = topic_batch:contiguous()
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  data.infos = infos

  return data
end

