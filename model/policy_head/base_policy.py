import torch
import torch.nn as nn
from einops import rearrange


def lstm_decoder(
    in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float
) -> torch.nn.Module:
    return nn.LSTM(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=True,
        dropout=policy_rnn_dropout_p,
    )


class MLPTanhHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPNohHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.mlp(x)

class MLPSigmoidHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)
    

class BasePolicyHead(nn.Module):

    def __init__(self, hidden_size, action_dim, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
    
    @staticmethod
    def _get_target_modal_tokens(tok_seq, tok_mask):
        index = tok_mask.nonzero(as_tuple=True)
        return tok_seq[index]
    
    def get_modal_tokens(self, tok_seq, tok_mask_dict, modal_name):
        assert modal_name in tok_mask_dict, f"{modal_name} not in token sequence"
        return self._get_target_modal_tokens(tok_seq, tok_mask_dict[modal_name])
    
    def loss(self, pred_action, labels, attention_mask=None):
        """
        pred_action: [bs, seq_len, chunck_size, 7], 1-6 refers to ee pose, 7 refers to gripper open/close
        lables: (pose gt [bs, seq_len, chunck_size, 6], gripper gt [bs, seq_len, chunck_size])
        attention_mask: [bs, seq_len, chunck_size]
        """
        if labels is None:
            return {"loss": None}
        # import pdb; pdb.set_trace()
        if isinstance(pred_action, tuple) or isinstance(pred_action, list):
            pred_action = torch.cat([pred_action[0], pred_action[1].unsqueeze(-1)], dim=-1)
        if attention_mask is None:
            pose_loss = torch.nn.functional.huber_loss(pred_action[..., :6], labels[0])
            gripper_loss = torch.nn.functional.binary_cross_entropy(pred_action[..., -1], labels[1])
        else:
            pose_loss = torch.nn.functional.huber_loss(pred_action[..., :6], labels[0], reduction='none')
            pose_loss = (pose_loss * attention_mask.unsqueeze(-1)).mean()
            gripper_loss = torch.nn.functional.binary_cross_entropy(pred_action[..., -1], labels[1], reduction='none')
            gripper_loss = (gripper_loss * attention_mask).mean()
        
        gripper_action_preds = (pred_action[..., -1] > 0.5).float()
        acc_gripper_act = torch.eq(gripper_action_preds, labels[1]).float()
        if attention_mask is None:
            acc_gripper_act = acc_gripper_act.mean()
        else:
            acc_gripper_act = (acc_gripper_act * attention_mask).sum() / attention_mask.sum()
        
        return {"loss_arm": pose_loss, "loss_gripper": gripper_loss, "acc_gripper": acc_gripper_act.item()}


class FCDecoder(BasePolicyHead):

    def __init__(self, hidden_size, action_dim, down_sample, latent, fwd_pred_next_n, **kwargs):
        super(FCDecoder, self).__init__(hidden_size, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.fwd_pred_next_n = fwd_pred_next_n
        self.actions = MLPTanhHead(self.hidden_size, fwd_pred_next_n * self.action_dim-1)
        self.gripper = MLPSigmoidHead(self.hidden_size, fwd_pred_next_n)
        if self.down_sample == 'pooling':
            self.pooling = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == 'resampler':
            pass
        else:
            raise NotImplementedError

    def forward(self, tok_seq):
        if self.down_sample == 'pooling':
            tok_seq = self.pooling(tok_seq.permute(0, 1, 3, 2)) # bs, seq_len, n_tok, tok_dim -> bs, seq_len, tok_dim, n_tok
            tok_seq = rearrange(tok_seq, 'b l d n -> b l (n d)')
            
        elif self.down_sample == 'resampler':
            pass
        else:
            raise NotImplementedError

        actions = self.actions(tok_seq)
        gripper = self.gripper(tok_seq)
        
        action = rearrange(action, 'b l (n d) -> b l n d', n=self.fwd_pred_next_n)
        gripper = rearrange(gripper, 'b l (n d) -> b l n d', n=self.fwd_pred_next_n)

        return actions, gripper


class LSTMDecoder(BasePolicyHead):

    def __init__(self, in_features, action_dim, down_sample, latent, fwd_pred_next_n, window_size, 
                 hidden_size=1024, num_layers=4, policy_rnn_dropout_p=0.1, **kwargs):
        super(LSTMDecoder, self).__init__(in_features, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.history_memory = []
        self.hidden_size = hidden_size
        self.rnn = lstm_decoder(in_features, hidden_size*latent, num_layers, policy_rnn_dropout_p)
        self.actions = MLPTanhHead(self.hidden_size, fwd_pred_next_n * (self.action_dim-1))
        self.gripper = MLPSigmoidHead(self.hidden_size, fwd_pred_next_n)
        self.hidden_state = None
        if self.down_sample == 'pooling':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == 'resampler':
            pass
        else:
            raise NotImplementedError
    
    def reset(self):
        self.hidden_state = None
        self.history_memory = []
    
    def forward(self, tok_seq, h_0=None):
        
        if self.down_sample == 'pooling':
            bs, seq_len = tok_seq.shape[:2]
            tok_seq = rearrange(tok_seq, 'b l n d-> (b l) n d')
            tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1)) # bs*seq_len, n_tok, tok_dim -> bs*seq_len, tok_dim, n_tok
            tok_seq = rearrange(tok_seq, '(b l) d n -> b l (n d)', b=bs, l=seq_len)
        elif self.down_sample == 'resampler':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        if tok_seq.shape[1] == 1:
            self.history_memory.append(tok_seq)
            if len(self.history_memory) <= self.history_len:
                # print('cur hist_mem len: {}'.format(len(self.history_memory)))
                x, h_n = self.rnn(tok_seq, self.hidden_state)
                self.hidden_state = h_n
                x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
            else:
                # the hidden state need to be refreshed based on the history window
                # print('hist_mem exceeded, refresh hidden state')
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                self.hidden_state = None
                x, h_n = self.rnn(hist_feature, self.hidden_state)
                x = x[:, -1].unsqueeze(1)
        else:
            self.hidden_state = h_0
            x, h_n = self.rnn(tok_seq, self.hidden_state)
            self.hidden_state = h_n

        actions = self.actions(x)
        gripper = self.gripper(x)

        actions = rearrange(actions, 'b l (n d) -> b l n d', n=self.fwd_pred_next_n)
        # gripper = rearrange(gripper, 'b l (n d) -> b l n', n=self.fwd_pred_next_n)

        return actions, gripper

if __name__ == '__main__':
    net = LSTMDecoder(
        in_features=1024, action_dim=7, down_sample='pooling', latent=1, fwd_pred_next_n=2, window_size=12,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    bs = 5
    window_size=12
    text_len = 8
    tokens = torch.randn(bs, window_size, text_len, 1024)
    # actions, gripper = net(tokens)
    # pred_actions = torch.cat([actions, gripper.unsqueeze(-1)], dim=-1)
    labels = (torch.randn(bs, window_size, 2, 6), torch.ones(bs, window_size, 2))
    att_mask = torch.ones(bs, window_size, 2)
    for i in range(10000):
        actions, gripper = net(tokens)
        pred_actions = torch.cat([actions, gripper.unsqueeze(-1)], dim=-1)
        optimizer.zero_grad()
        loss = net.loss(pred_actions, labels, att_mask)
        
        loss_arm = loss['loss_arm']; loss_gripper = loss['loss_gripper']; acc_gripper = loss['acc_gripper']
        loss_act = loss_arm + 0.01 * loss_gripper
        loss_act.backward()
        optimizer.step()
        print("iter: {}, loss: {} gripper: {} acc: {}".format(i, loss_act.item(), loss_gripper.item(), acc_gripper))
    print(loss)
    pass