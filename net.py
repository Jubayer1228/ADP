import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
import torch.distributions as dist
from transformers import GPT2Config, GPT2Model
from transformers import ReformerConfig, ReformerModel
from transformers import TransfoXLConfig, TransfoXLModel
from IPython import embed
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=1,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]

        state_seq = torch.cat([query_states, x['context_states']], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions']], dim=1)
        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states']], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards']], dim=1)

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]
    


class Transformer1(nn.Module):
    """Transformer with integrated dopamine-based exploration bonus and memory for good events."""
    
    def __init__(self, config):
        super(Transformer1, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = config['horizon']
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.dropout = config['dropout']

        # Original GPT2 configuration.
        gpt2_config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=1,  # Adjust as needed.
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
            integrate_bonus=True,
        )
        self.transformer = GPT2Model(gpt2_config)

        # Embedding layer for transitions.
        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        # Action prediction layer.
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        
        # Bonus prediction branch: predicts a scalar bonus per time step.
        self.pred_bonus = nn.Linear(self.n_embd, 1)
        
        # Bonus integration: projects the bonus signal into the embedding space.
        self.bonus_integration = nn.Linear(1, self.n_embd)
        
        # Memory for "good events" (optional).
        self.good_event_memory = []  # list to store good events
        self.good_event_threshold = config.get('good_event_threshold', 1.0)

        # Hyperparameters.
        self.bonus_scale = config.get('bonus_scale', 0.1)  # scales uncertainty bonus
        self.integrate_bonus = config.get('integrate_bonus', True)  # if True, modulate hidden states

        # Epoch-level counters for tracking "good events".
        self.epoch_good_count = 0
        self.epoch_total_count = 0

    def forward(self, x):
        """
        Args:
            x (dict): Contains:
                - 'query_states': Tensor (B, state_dim)
                - 'context_states': Tensor (B, seq_len, state_dim)
                - 'context_actions': Tensor (B, seq_len, action_dim)
                - 'context_next_states': Tensor (B, seq_len, state_dim)
                - 'context_rewards': Tensor (B, seq_len, 1)
                - 'zeros': Helper tensor for padding.
        
        Returns:
            If self.test is False (training): predicted actions tensor.
            If self.test is True (evaluation): tuple (last action prediction, bonus prediction, dopamine bonus).
        """
        # Build the sequence.
        query_states = x['query_states'][:, None, :]  # (B, 1, state_dim)
        zeros = x['zeros'][:, None, :]

        state_seq = torch.cat([query_states, x['context_states']], dim=1)
        action_seq = torch.cat([zeros[:, :, :self.action_dim], x['context_actions']], dim=1)
        next_state_seq = torch.cat([zeros[:, :, :self.state_dim], x['context_next_states']], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards']], dim=1)

        # Concatenate state, action, next state, and reward information.
        seq = torch.cat([state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)  # (B, seq_len, n_embd)

        # Run the transformer.
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        hidden_states = transformer_outputs['last_hidden_state']  # (B, seq_len, n_embd)

        # --- Bonus branch computations ---
        bonus_pred = self.pred_bonus(hidden_states)  # (B, seq_len, 1)
        rpe = F.relu(reward_seq - bonus_pred)          # (B, seq_len, 1)
        pred_prob = torch.sigmoid(bonus_pred)           # in [0, 1]
        uncertainty = 1 - 4 * torch.abs(pred_prob - 0.5)
        uncertainty = F.relu(uncertainty)
        dopamine_bonus = rpe + self.bonus_scale * uncertainty  # (B, seq_len, 1)
        # --- End bonus branch computations ---

        # --- Bonus integration into action prediction ---
        if self.integrate_bonus:
            bonus_gate = torch.sigmoid(self.bonus_integration(bonus_pred))  # (B, seq_len, n_embd)
            modulated_hidden = hidden_states * (1 + self.bonus_scale * bonus_gate)
        else:
            modulated_hidden = hidden_states

        preds = self.pred_actions(modulated_hidden)  # (B, seq_len, action_dim)
        # --- End bonus integration ---

        # --- Track and save "good events" during training ---
        if not self.test:
            batch_size, seq_len, _ = reward_seq.shape
            num_good = 0
            for b in range(batch_size):
                if dopamine_bonus[b, -1, 0] > self.good_event_threshold:
                    event = {
                        'state': state_seq[b, -1, :].detach().cpu(),
                        'action': action_seq[b, -1, :].detach().cpu(),
                        'reward': reward_seq[b, -1, :].detach().cpu(),
                        'dopamine_bonus': dopamine_bonus[b, -1, 0].detach().cpu()
                    }
                    self.good_event_memory.append(event)
                    num_good += 1
            # Update epoch-level counters.
            self.epoch_good_count += num_good
            self.epoch_total_count += batch_size
            # Optionally, you can print batch-level metrics as well:
            batch_percentage = (num_good / batch_size) * 100
            print(f"Batch good events: {batch_percentage:.2f}%")
        # --- End good events tracking ---

        if self.test:
            return preds[:, -1, :], bonus_pred[:, -1, :], dopamine_bonus[:, -1, :]
        return preds[:, 1:, :]

    def report_epoch_good_events(self):
        """Call this at the end of an epoch to report the percentage of good events."""
        if self.epoch_total_count > 0:
            percentage_good = (self.epoch_good_count / self.epoch_total_count) * 100
        else:
            percentage_good = 0.0
        print(f"Epoch good events: {percentage_good:.2f}%")
        # Optionally, reset counters for the next epoch.
        self.epoch_good_count = 0
        self.epoch_total_count = 0
        return percentage_good


# Example usage:
# config = {
#     'test': False,
#     'horizon': 100,
#     'n_embd': 128,
#     'n_layer': 4,
#     'n_head': 4,
#     'state_dim': 5,
#     'action_dim': 3,
#     'dropout': 0.1,
#     'good_event_threshold': 1.0,
#     'bonus_scale': 0.1,
#     'integrate_bonus': True,
# }
# model = TransformerDopamine(config)






class ImageTransformer(Transformer):
    """Transformer class for image-based data."""

    def __init__(self, config):
        super().__init__(config)
        self.im_embd = 8

        size = self.config['image_size']
        size = (size - 3) // 2 + 1
        size = (size - 3) // 1 + 1

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        new_dim = self.im_embd + self.state_dim + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)

    def forward(self, x):
        query_images = x['query_images'][:, None, :]
        query_states = x['query_states'][:, None, :]
        context_images = x['context_images']
        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        batch_size = query_states.shape[0]

        image_seq = torch.cat([query_images, context_images], dim=1)
        image_seq = image_seq.view(-1, *image_seq.size()[2:])

        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)

        context_states = torch.cat([query_states, context_states], dim=1)
        context_actions = torch.cat([
            torch.zeros(batch_size, 1, self.action_dim).to(device),
            context_actions,
        ], dim=1)
        context_rewards = torch.cat([
            torch.zeros(batch_size, 1, 1).to(device),
            context_rewards,
        ], dim=1)

        stacked_inputs = torch.cat([
            image_enc_seq,
            context_states,
            context_actions,
            context_rewards,
        ], dim=2)
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]