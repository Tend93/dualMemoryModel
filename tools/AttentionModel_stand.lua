require 'nn'
require 'nngraph'

local Attention = {}
local dropout = 0
if dropout>0 then print('attention dropout is',dropout) end
-- feat for selector, feat_size 
-- att_seq for candidates, seq_size, amount - att_size 
function Attention.baseModel(feat,feat_size,att_seq,seq_size,att_size, att_hid_size)
    local att = nn.View(-1, seq_size)(att_seq)         -- (batch * att_size) * feat_size
    local att_h, dot
    print('base attention model...')
    if att_hid_size > 0 then
        att = nn.Linear(seq_size, att_hid_size)(att)        -- (batch * att_size) * att_hid_size
        att = nn.View(-1, att_size, att_hid_size)(att)      -- batch * att_size * att_hid_size

        att_h = nn.Linear(feat_size, att_hid_size)(feat)    -- batch * att_hid_size
        att_h = nn.Replicate(att_size, 2)(att_h)            -- batch * att_size * att_hid_size

        dot = nn.CAddTable(){att_h, att}                    -- batch * att_size * att_hid_size
        dot = nn.Tanh()(dot)                                -- batch * att_size * att_hid_size
        if dropout>0 then dot = nn.Dropout(dropout)(dot) end
	    dot = nn.View(-1, att_hid_size)(dot)                -- (batch * att_size) * att_hid_size
        dot = nn.Linear(att_hid_size, 1)(dot)               -- (batch * att_size) * 1
        dot = nn.View(-1, att_size)(dot)                    -- batch * att_size
    else
        att = nn.Linear(seq_size, 1)(att)                   -- (batch * att_size) * 1
        att = nn.View(-1, att_size)(att)                    -- batch * att_size
        att_h = nn.Linear(feat_size, 1)(feat)               -- batch * 1
        att_h = nn.Replicate(att_size, 2)(att_h)            -- batch * att_size * 1
        att_h = nn.Squeeze()(att_h)                         -- batch * att_size
        dot = nn.CAddTable(){att_h, att}                    -- batch * att_size
    end

    local weight = nn.SoftMax()(dot)

    local att_res = nn.MixtureTable(2){weight, att_seq}   -- batch * feat_size <- (batch * att_size, batch * feat_size * att_size)
    return att_res
end

function Attention.baseModel_alpha(feat,feat_size,att_seq,seq_size,att_size, att_hid_size)
    local att = nn.View(-1, seq_size)(att_seq)         -- (batch * att_size) * feat_size
    local att_h, dot
    print('base attention model...')
    if att_hid_size > 0 then
        att = nn.Linear(seq_size, att_hid_size)(att)        -- (batch * att_size) * att_hid_size
        att = nn.View(-1, att_size, att_hid_size)(att)      -- batch * att_size * att_hid_size

        att_h = nn.Linear(feat_size, att_hid_size)(feat)    -- batch * att_hid_size
        att_h = nn.Replicate(att_size, 2)(att_h)            -- batch * att_size * att_hid_size

        dot = nn.CAddTable(){att_h, att}                    -- batch * att_size * att_hid_size
        dot = nn.Tanh()(dot)                                -- batch * att_size * att_hid_size
        if dropout>0 then dot = nn.Dropout(dropout)(dot) end
        dot = nn.View(-1, att_hid_size)(dot)                -- (batch * att_size) * att_hid_size
        dot = nn.Linear(att_hid_size, 1)(dot)               -- (batch * att_size) * 1
        dot = nn.View(-1, att_size)(dot)                    -- batch * att_size
    else
        att = nn.Linear(seq_size, 1)(att)                   -- (batch * att_size) * 1
        att = nn.View(-1, att_size)(att)                    -- batch * att_size
        att_h = nn.Linear(feat_size, 1)(feat)               -- batch * 1
        att_h = nn.Replicate(att_size, 2)(att_h)            -- batch * att_size * 1
        att_h = nn.Squeeze()(att_h)                         -- batch * att_size
        dot = nn.CAddTable(){att_h, att}                    -- batch * att_size
    end

    local weight = nn.SoftMax()(dot)

    local att_res = nn.MixtureTable(2){weight, att_seq}   -- batch * feat_size <- (batch * att_size, batch * feat_size * att_size)
    return att_res, weight
end

function Attention.baseModel_dot(feat, feat_size, att_seq, seq_size, att_size)
    print('dot attention model...')
	local att = nn.View(-1, seq_size)(att_seq)
	local att2feat = nn.Linear(seq_size, feat_size)(att)
	att2feat = nn.View(-1, att_size, feat_size)(att2feat)
	
	local att_h = nn.Replicate(att_size,2)(feat)
	local concat = nn.CMulTable()({att2feat, att_h})
	local score = nn.Sum(3)(concat)
	score = nn.Squeeze()(score)
	local weight = nn.SoftMax()(score)
	
    local att_res = nn.MixtureTable(2){weight, att_seq}
	return att_res
end

return Attention
