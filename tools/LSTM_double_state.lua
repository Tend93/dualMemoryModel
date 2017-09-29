require 'nn'
require 'nngraph'
local Attention = require 'tools.AttentionModel_stand'

local LSTM_tools = {}
print('file version LSTM tools')
local dropout = 0.8
-------------------------------------------------------------------------
-- rnn_size for size of LSTM hidden state and cell state
-- input_size for size of input x
-- 
-------------------------------------------------------------------------
function LSTM_tools.lstm(opt)
    -- Model parameters
    local rnn_size = opt.rnn_size 
    local input_size = opt.input_size
    print('.......................lstm parameters..............')
    print(opt)
    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------- LSTM main part --------------------
    function buildGate(in_size, xf, prev_hf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
        local h2h = nn.Linear(rnn_size, rnn_size,false)(prev_hf)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate = nn.Sigmoid()(buildGate(input_size, x, prev_h))
    local forget_gate = nn.Sigmoid()(buildGate(input_size, x, prev_h))
    local out_gate = nn.Sigmoid()(buildGate(input_size, x, prev_h))
    local in_transform = nn.Tanh()(buildGate(input_size, x, prev_h))

    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    
    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})

end

function LSTM_tools.lstm_soft_att_double_state(opt)
    -- Model parameters 
    -- for input feat and lstm
    local rnn_size = opt.rnn_size
    local input_size = opt.input_size
    -- for attention seq
    local feat_size = opt.att_feat_size        -- for attention feats size
    local att_size = opt.att_length            -- for amount of attention feats
    local att_hid_size = opt.att_hid_size      -- for attention hidden size
    local att_type = opt.att_type or 3
    print('.......................LSTM Soft Attention with Double State Parameters..............')
    print(opt)
	local inputs = {}
	local outputs = {}
	for i=1,6 do
		table.insert(inputs,nn.Identity()())
	end
    local x = inputs[1]         -- batch * input_size -- embedded caption at a specific step
    local att_seq = inputs[2]   -- batch * att_size * feat_size -- the image patches
    local prev_c = inputs[3]
    local prev_h = inputs[4]
	local local_prev_c = inputs[5]
    local local_prev_h = inputs[6]
    ------------ Attention part --------------------
    local att_res 
    if att_type == 1 then
        print('state attention with local_prev_h')
        att_res = Attention.baseModel(local_prev_h,rnn_size,att_seq, feat_size, att_size,att_hid_size)
    elseif att_type == 2 then
        print('state attention with prev_h')
        att_res = Attention.baseModel(prev_h,rnn_size,att_seq, feat_size, att_size,att_hid_size)
    elseif att_type == 3 then
        print('state attention with content')
        att_res = Attention.baseModel(x,input_size,att_seq, feat_size, att_size,att_hid_size)
    end
    ------------ End of attention part -----------

    ------------- LSTM main part --------------------
    function buildState(in_size, xf, prev_hf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
        local h2h = nn.Linear(rnn_size, rnn_size, false)(prev_hf)
        return nn.CAddTable()({i2h, h2h})
    end
	
	function buildGate(in_size, local_in_size, xf,loca_xf, prev_hf,local_prev_hf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
		local i2hlocal = nn.Linear(local_in_size, rnn_size,false)(loca_xf)
        local h2h = nn.Linear(rnn_size, rnn_size, false)(prev_hf)
		local h2hlocal = nn.Linear(rnn_size, rnn_size, false)(local_prev_hf)
        return nn.CAddTable()({i2h,i2hlocal,h2h,h2hlocal})
    end

    local in_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local forget_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local out_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local in_transform = nn.Tanh()(buildState(input_size, x, prev_h))
	local local_in_transform = nn.Tanh()(buildState(feat_size, att_res, local_prev_h))
	
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })	
	local local_next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, local_prev_c}),
        nn.CMulTable()({in_gate,     local_in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    local local_next_h = nn.CMulTable()({out_gate, nn.Tanh()(local_next_c)})
	table.insert(outputs,next_c)
	table.insert(outputs,next_h)
	table.insert(outputs,local_next_c)
	table.insert(outputs,local_next_h)
    return nn.gModule(inputs, outputs)
end

function LSTM_tools.lstm_soft_att_double_state_alpha(opt)
    -- Model parameters 
    -- for input feat and lstm
    local rnn_size = opt.rnn_size
    local input_size = opt.input_size
    -- for attention seq
    local feat_size = opt.att_feat_size
    local att_size = opt.att_length
    local att_hid_size = opt.att_hid_size
    local att_type = opt.att_type or 3
    print('.......................LSTM Soft Attention with Double State Out_alpha parameters..............')
    print(opt)
    local inputs = {}
    local outputs = {}
    for i=1,6 do
        table.insert(inputs,nn.Identity()())
    end
    local x = inputs[1]         -- batch * input_size -- embedded caption at a specific step
    local att_seq = inputs[2]   -- batch * att_size * feat_size -- the image patches
    local prev_c = inputs[3]
    local prev_h = inputs[4]
    local local_prev_c = inputs[5]
    local local_prev_h = inputs[6]
    ------------ Attention part --------------------
    local att_res, alpha 
    if att_type == 1 then
        print('state attention with local_prev_h')
        att_res, alpha = Attention.baseModel_alpha(local_prev_h,rnn_size,att_seq, feat_size, att_size,att_hid_size)
    elseif att_type == 2 then
        print('state attention with prev_h')
        att_res, alpha = Attention.baseModel_alpha(prev_h,rnn_size,att_seq, feat_size, att_size,att_hid_size)
    elseif att_type == 3 then
        print('state attention with content')
        att_res, alpha = Attention.baseModel_alpha(x,input_size,att_seq, feat_size, att_size,att_hid_size)
    end
    ------------ End of attention part -----------

    ------------- LSTM main part --------------------
    function buildState(in_size, xf, prev_hf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
        local h2h = nn.Linear(rnn_size, rnn_size, false)(prev_hf)
        return nn.CAddTable()({i2h, h2h})
    end
    
    function buildGate(in_size, local_in_size, xf,loca_xf, prev_hf,local_prev_hf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
        local i2hlocal = nn.Linear(local_in_size, rnn_size,false)(loca_xf)
        local h2h = nn.Linear(rnn_size, rnn_size, false)(prev_hf)
        local h2hlocal = nn.Linear(rnn_size, rnn_size, false)(local_prev_hf)
        return nn.CAddTable()({i2h,i2hlocal,h2h,h2hlocal})
    end

    local in_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local forget_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local out_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local in_transform = nn.Tanh()(buildState(input_size, x, prev_h))
    local local_in_transform = nn.Tanh()(buildState(feat_size, att_res, local_prev_h))
    
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })  
    local local_next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, local_prev_c}),
        nn.CMulTable()({in_gate,     local_in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    local local_next_h = nn.CMulTable()({out_gate, nn.Tanh()(local_next_c)})
    table.insert(outputs,next_c)
    table.insert(outputs,next_h)
    table.insert(outputs,local_next_c)
    table.insert(outputs,local_next_h)
    table.insert(outputs,alpha)
    return nn.gModule(inputs, outputs)
end

function LSTM_tools.lstm_soft_att_double_state_out(opt)
    -- Model parameters 
    -- for input feat and lstm
    local rnn_size = opt.rnn_size
    local input_size = opt.input_size
    -- for attention seq
    local feat_size = opt.att_feat_size
    local att_size = opt.att_length
    local topic_size = opt.topic_size
    local att_hid_size = opt.att_hid_size
    local att_type = opt.att_type or 3
    print('.......................LSTM Soft Attention with Double State Out_topic_feat parameters..............')
    print(opt)
    local inputs = {}
    local outputs = {}
    for i=1,6 do
        table.insert(inputs,nn.Identity()())
    end
    local x = inputs[1]         -- batch * input_size -- embedded caption at a specific step
    local att_seq = inputs[2]   -- batch * att_size * feat_size -- the image patches
    local prev_c = inputs[3]
    local prev_h = inputs[4]
    local local_prev_c = inputs[5]
    local local_prev_h = inputs[6]
    ------------ Attention part --------------------
    local att_res 
    if att_type == 1 then
        print('state attention with local_prev_h')
        att_res = Attention.baseModel(local_prev_h,rnn_size,att_seq, feat_size, att_size,att_hid_size)
    elseif att_type == 2 then
        print('state attention with prev_h')
        att_res = Attention.baseModel(prev_h,rnn_size,att_seq, feat_size, att_size,att_hid_size)
    elseif att_type == 3 then
        print('state attention with content')
        att_res = Attention.baseModel(x,input_size,att_seq, feat_size, att_size,att_hid_size)
    end
    ------------ End of attention part -----------
    local att_out = nn.Linear(feat_size, topic_size)(att_res)
    ------------- LSTM main part --------------------
    function buildState(in_size, xf, prev_hf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
        local h2h = nn.Linear(rnn_size, rnn_size, false)(prev_hf)
        return nn.CAddTable()({i2h, h2h})
    end
    
    function buildGate(in_size, local_in_size, xf,loca_xf, prev_hf,local_prev_hf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
        local i2hlocal = nn.Linear(local_in_size, rnn_size,false)(loca_xf)
        local h2h = nn.Linear(rnn_size, rnn_size, false)(prev_hf)
        local h2hlocal = nn.Linear(rnn_size, rnn_size, false)(local_prev_hf)
        return nn.CAddTable()({i2h,i2hlocal,h2h,h2hlocal})
    end

    local in_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local forget_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local out_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, att_res,prev_h,local_prev_h))
    local in_transform = nn.Tanh()(buildState(input_size, x, prev_h))
    local local_in_transform = nn.Tanh()(buildState(feat_size, att_res, local_prev_h))
    
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })  
    local local_next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, local_prev_c}),
        nn.CMulTable()({in_gate,     local_in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    local local_next_h = nn.CMulTable()({out_gate, nn.Tanh()(local_next_c)})
    table.insert(outputs,next_c)
    table.insert(outputs,next_h)
    table.insert(outputs,local_next_c)
    table.insert(outputs,local_next_h)
    table.insert(outputs, att_out)
    return nn.gModule(inputs, outputs)
end

function LSTM_tools.lstm_language_att(opt)
    -- Model parameters
    local rnn_size = opt.rnn_size
    local input_size = opt.input_size
    -- for attention seq
    local feat_size = opt.att_feat_size
    local att_hid_size = opt.att_hid_size
    local att_size = opt.att_length
    local output_size = opt.output_size     -- for word prediction
    print('.......................LSTM Language Model with attention parameters..............')
    print(opt)
    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local att_seq = nn.Identity()()   -- batch * att_size * feat_size -- the image patches
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------ Attention part --------------------
    local att_res = Attention.baseModel(prev_h,rnn_size,att_seq,feat_size, att_size, att_hid_size)

    ------------ End of attention part -----------

    ------------- LSTM main part --------------------
    function buildGate(in_size,f_size, xf, prev_hf, att_resf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
        local h2h = nn.Linear(rnn_size, rnn_size, false)(prev_hf)
        local att_add = nn.Linear(f_size, rnn_size, false)(att_resf)
        return nn.CAddTable()({i2h, h2h,att_add})
    end

    local in_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, prev_h, att_res))
    local forget_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, prev_h, att_res))
    local out_gate = nn.Sigmoid()(buildGate(input_size,feat_size, x, prev_h, att_res))
    local in_transform = nn.Tanh()(buildGate(input_size,feat_size, x, prev_h, att_res))

    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    local dropped = nn.Dropout(dropout)(next_h)
    local proj = nn.Linear(rnn_size, output_size)(dropped)
    local logsoft = nn.LogSoftMax()(proj)

    return nn.gModule({x, att_seq, prev_c, prev_h}, {next_c, next_h, logsoft})

end

function LSTM_tools.lstm_language(opt)
    -- Model parameters
    local rnn_size = opt.rnn_size
    local input_size = opt.input_size
    local output_size = opt.output_size
    print('.......................LSTM Language Model parameters..............')
    print(opt)
    local x = nn.Identity()()         -- batch * input_size -- embedded caption at a specific step
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    ------------- LSTM main part --------------------
    function buildGate(in_size, xf, prev_hf)
        local i2h = nn.Linear(in_size, rnn_size)(xf)
        local h2h = nn.Linear(rnn_size, rnn_size, false)(prev_hf)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate = nn.Sigmoid()(buildGate(input_size, x, prev_h))
    local forget_gate = nn.Sigmoid()(buildGate(input_size, x, prev_h))
    local out_gate = nn.Sigmoid()(buildGate(input_size, x, prev_h))
    local in_transform = nn.Tanh()(buildGate(input_size, x, prev_h))

    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) -- batch * rnn_size
    local dropped = nn.Dropout(dropout)(next_h)
    local proj = nn.Linear(rnn_size, output_size)(dropped)
    local logsoft = nn.LogSoftMax()(proj)

    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h, logsoft})
end

return LSTM_tools

