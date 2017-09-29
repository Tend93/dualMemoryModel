local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

function utils.shuffle_table(t)
  assert(t, "shuffle_table() expected a table, got nil")
  math.randomseed(os.time())
  local j
  for i = #t, 2, -1 do
    j = math.random(i)
    t[i], t[j] = t[j], t[i]
  end
end

function utils.shuffle_tables(t1, t2)
  assert(t1, "shuffle_table() expected a table, got nil")
  assert(#t1 == #t2, 'tables should have the same size...')
  math.randomseed(os.time())
  local j
  for i = #t1, 2, -1 do
    j = math.random(i)
    t1[i], t1[j] = t1[j], t1[i]
    t2[i], t2[j] = t2[j], t2[i]
  end
end

function utils.expand_feat(t, batch_size, seq_per_img, gpuid)
  local newt = torch.Tensor(batch_size * seq_per_img, t:size(2), t:size(3))
  if gpuid >= 0 then newt = newt:cuda() end
  for x = 1, batch_size do
    for y = 1, seq_per_img do
      newt[(x-1)*seq_per_img + y] = t[x]
    end
  end
  return newt
end

function utils.merge2table(t1, t2)
  for k, v in pairs(t2) do
    table.insert(t1, v)
  end
  return t1
end

return utils
