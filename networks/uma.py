
'''
Author: FnoY fangying@westlake.edu.cn
LastEditTime: 2024-10-08 17:50:13
FilePath: /InASR/networks/uma.py
Notes: If the feature dimension changes from 256 to 512, just modify 'output_size: int = 256' to 'output_size: int = 512'.
'''
# """Unimodal aggregation definition."""
import logging
from typing import Optional, Tuple
import torch
# from typeguard import typechecked
from networks.transformer.layer_norm import LayerNorm
from utils.nets_utils import get_activation


class UMA(torch.nn.Module):
    """UMA module.

    """

    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 256,
        lookahead_kernel: int = 0,
        right_context: int = 0,
    ):
        # assert typechecked()
        super().__init__()
        self._output_size = output_size
        input_size = output_size
        self.chunk_wize = False

        if lookahead_kernel > 0:
            self.right_context = right_context
            self.left_context = lookahead_kernel - 1 - self.right_context
            # self.depthwise_conv = nn.Conv1d(
            #     output_size,
            #     output_size,
            #     lookahead_kernel,
            #     stride=1,
            #     padding=lookahead_kernel // 2,
            #     groups=output_size,
            #     bias=True,
            # )

            self.lookahead_cnn_uma = torch.nn.Conv1d(
                output_size,
                output_size,
                lookahead_kernel,
                stride=1,
                padding=0,
                bias=True,
            )

            # self.lookahead_cnn =  nn.Conv1d(
            #         output_size,
            #         256,
            #         lookahead_kernel // 2 +1,
            #         stride=1,
            #         padding=lookahead_kernel // 4,
            #         bias=True
            #     )

            activation_type = "swish"
            self.activation = get_activation(activation_type)
            self.dropout = torch.nn.Dropout(0.1)
   
            # self.lookahead_cnn2 = nn.Conv1d(
            #         256,
            #         256,
            #         lookahead_kernel // 2 +1,
            #         stride=1,
            #         padding=lookahead_kernel // 4,
            #         bias=True
            #     )

            self.lookahead_norm = LayerNorm(256)

        self.linear_sigmoid = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1),
            torch.nn.Sigmoid(),
        )

        self.after_norm = LayerNorm(input_size)

    def output_size(self) -> int:
        return self._output_size


    def forward(
        self,
        xs_pad: torch.Tensor,
        olens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.training:
            return self.forward_train(xs_pad, olens)
        else:
            return self.forward_infer(xs_pad, olens, prev_states)
        

    def forward_train(
        self,
        xs_pad: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            olens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
        Returns:
            torch.Tensor: Output tensor (#batch, I, output_size).  
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """
        chunk_counts = None
        # logging.info(f'xspad:{xs_pad.shape}, olens:{olens}')
        batch, length, _ = xs_pad.size()
        # xs_pad = self.before_norm(xs_pad)
        # uma_weights = self.gen_uma(xs_pad)

        if hasattr(self, "lookahead_cnn_uma"):
            xs_pad2 = torch.nn.functional.pad(xs_pad, (0, 0, self.left_context, self.right_context))
            xs_pad2 = self.lookahead_cnn_uma(xs_pad2.transpose(1, 2)).transpose(1, 2)
            xs_pad2 = self.activation(xs_pad2)
            xs_pad2 = self.dropout(xs_pad2)
            xs_pad2 = self.lookahead_norm(xs_pad2)
            uma_weights = self.linear_sigmoid(xs_pad2)
        else:
            uma_weights = self.linear_sigmoid(xs_pad)

        # Unimodal Detection
        scalar_before = uma_weights[:,:-1,:].detach() # (#batch, L-1, 1)
        scalar_after = uma_weights[:,1:,:].detach() # (#batch, L-1, 1)
        scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))    # (#batch, L, 1)
        scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))  # (#batch, L, 1)

        mask = (uma_weights.lt(scalar_before)) & (uma_weights.lt(scalar_after)) # bool tensor (#batch, L, 1)
        mask = mask.reshape(uma_weights.shape[0], -1) # bool tensor (#batch, L)

        if self.chunk_wize:
            # chunkwise
            # logging.info(f'length:{length}, mask.device:{mask.device}')
            positions = (24 + 16 * torch.arange((length-24)//16)).to(mask.device)
            # logging.info(f'positions:{positions}')
            mask[:, positions] = True

        mask[:,0] = True

        if self.chunk_wize:
            # chunkwise
            start_idx = torch.cat((torch.tensor([0]).to(positions.device), positions), dim=0)
            end_idx = torch.cat((positions, torch.tensor([length]).to(positions.device)), dim=0)
            # chunk_counts = mask[:,start_idx:end_idx].sum(dim=-1)
            split_sizes = end_idx - start_idx
            # logging.info(f'split_sizes:{split_sizes}')
            chunks = torch.split(mask, split_sizes.tolist(), dim=1)
            chunk_counts = torch.stack([chunk.sum(dim=-1) for chunk in chunks]).T
            # logging.info(f'chunk_counts:{chunk_counts.sum(dim=-1)}, chunk_counts.shape:{chunk_counts.shape}')

        # mask.nonzero() is [[0,0],[0,3],[0,7],...,[1,0],[1,2],...,[2,0],[2,4],...,[#batch-1,0],...]
        # mask.nonzero() : (K,2); K is the total number of valleys in this batch
        batch_index = mask.nonzero()[:,0] # (k,1); [0,0,0,...,1,1,...,2,2,...,#batch-1,...]
        valley_index_start = mask.nonzero()[:,1] # (k,1); [0,3,7,...,0,2,...,0,4,...,0,...]
        mask[:,0] = False
        mask[:,-1] = True
        valley_index_end = mask.nonzero()[:,1] + 2 
        # (k,1); [5,9,...,4,...,6,...]
        valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end), 
                                       (length) * torch.ones_like(valley_index_end), valley_index_end)

        _,counts = torch.unique(batch_index, return_counts = True) # (#batch, 1); the number of valleys in each sample
        max_counts = (torch.max(counts)).item() 

        utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(xs_pad.device)
        batch_index_mask = utri_mat1[counts]
        batch_index_mask = batch_index_mask.reshape(-1,1)
        batch_index_mask = batch_index_mask.nonzero()[:, 0]

        valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        # logging.info(f'valleys:{valleys}')
        
        # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        utri_mat = torch.tril(torch.ones(length+1,length),-1).to(xs_pad.device)
        output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        output_mask = output_mask.detach()

        # Aggregation
        alpha_h = torch.mul(uma_weights, xs_pad)
        xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, uma_weights).clamp_(1e-6)

        xs_pad = self.after_norm(xs_pad)
        # olens = (olens / olens[0] * xs_pad.shape[1]).type_as(olens)
        olens = counts
        # logging.info(f'olens:{olens}')
        # return xs_pad, olens, (uma_weights, p_attn)
        return xs_pad, olens, chunk_counts
    
    
    def forward_infer(
        self,
        xs_pad: torch.Tensor,
        olens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            olens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
        Returns:
            torch.Tensor: Output tensor (#batch, I, output_size).  
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """
        chunk_counts = None
        # if prev_states is not None:
        #     logging.info(f'prev_states:{prev_states["n_processed_blocks"]}')
        # else:
        #     logging.info(f'prev_states is None')
        # logging.info(f'xspad:{xs_pad.shape}, olens:{olens}')
        batch, length, _ = xs_pad.size()
        # xs_pad = self.before_norm(xs_pad)
        # uma_weights = self.gen_uma(xs_pad)

        if hasattr(self, "lookahead_cnn_uma"):
            xs_pad2 = torch.nn.functional.pad(xs_pad, (0, 0, self.left_context, self.right_context))
            xs_pad2 = self.lookahead_cnn_uma(xs_pad2.transpose(1, 2)).transpose(1, 2)
            xs_pad2 = self.activation(xs_pad2)
            xs_pad2 = self.dropout(xs_pad2)
            xs_pad2 = self.lookahead_norm(xs_pad2)
            uma_weights = self.linear_sigmoid(xs_pad2)
        else:
            uma_weights = self.linear_sigmoid(xs_pad)

        # Unimodal Detection
        scalar_before = uma_weights[:,:-1,:].detach() # (#batch, L-1, 1)
        scalar_after = uma_weights[:,1:,:].detach() # (#batch, L-1, 1)
        scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))    # (#batch, L, 1)
        scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))  # (#batch, L, 1)

        mask = (uma_weights.lt(scalar_before)) & (uma_weights.lt(scalar_after)) # bool tensor (#batch, L, 1)
        
        # mask2 = (uma_weights.gt(scalar_before)) & (uma_weights.gt(scalar_after)) # bool tensor (#batch, L, 1)
        # mask = mask | mask2

        mask = mask.reshape(uma_weights.shape[0], -1) # bool tensor (#batch, L)

        if self.chunk_wize:
            # chunkwise
            # logging.info(f'length:{length}, mask.device:{mask.device}')
            positions = (24 + 16 * torch.arange((length-24)//16)).to(mask.device)
            # logging.info(f'positions:{positions}')
            mask[:, positions] = True

        mask[:,0] = True
        mask[:,-1] = False

        if self.chunk_wize:
            # chunkwise
            start_idx = torch.cat((torch.tensor([0]).to(positions.device), positions), dim=0)
            end_idx = torch.cat((positions, torch.tensor([length]).to(positions.device)), dim=0)
            # chunk_counts = mask[:,start_idx:end_idx].sum(dim=-1)
            split_sizes = end_idx - start_idx
            # logging.info(f'split_sizes:{split_sizes}')
            chunks = torch.split(mask, split_sizes.tolist(), dim=1)
            chunk_counts = torch.stack([chunk.sum(dim=-1) for chunk in chunks]).T
            # logging.info(f'chunk_counts:{chunk_counts.sum(dim=-1)}, chunk_counts.shape:{chunk_counts.shape}')
            
        # mask.nonzero() is [[0,0],[0,3],[0,7],...,[1,0],[1,2],...,[2,0],[2,4],...,[#batch-1,0],...]
        # mask.nonzero() : (K,2); K is the total number of valleys in this batch

        # mask_valley = mask.nonzero()
        # if prev_states is None:
        #     mask_valley = mask_valley[mask_valley[:,1]>8]
        #     length_min = 8
        #     length_max = 40
        # elif prev_states["n_processed_blocks"]==1:
        #     mask_valley = mask_valley[mask_valley[:,1]<24]
        #     length_min = 0
        #     length_max = 24
        # else:
        #     mask_valley = mask_valley[mask_valley[:,1]>8]
        #     mask_valley = mask_valley[mask_valley[:,1]<24]
        #     length_min = 8
        #     length_max = 24

        # batch_index = torch.cat([mask_valley[:,0],(torch.tensor([0]))]) # (k,1); [0,0,0,...,1,1,...,2,2,...,#batch-1,...]
        # valley_index_start = torch.cat([torch.tensor([length_min]),mask_valley[:,1]]) # (k,1); [0,3,7,...,0,2,...,0,4,...,0,...]
        # valley_index_end = torch.cat([mask_valley[:,1]+2,torch.tensor([length_max+1])]) # (k,1); [5,9,...,4,...,6,...]

        batch_index = mask.nonzero()[:,0] # (k,1); [0,0,0,...,1,1,...,2,2,...,#batch-1,...]
        valley_index_start = mask.nonzero()[:,1] # (k,1); [0,3,7,...,0,2,...,0,4,...,0,...]

        mask[:,0] = False
        mask[:,-1] = True
        valley_index_end = mask.nonzero()[:,1] + 2 
        
        # mask2 = (uma_weights.gt(scalar_before)) & (uma_weights.gt(scalar_after)) # bool tensor (#batch, L, 1)
        # mask2[:,0] = False
        # mask2[:,-1] = True
        # valley_index_end2 = mask2.nonzero()[:,1] + 2 
        # if valley_index_end.shape[0] > valley_index_end2.shape[0]:
        #     valley_index_end[1:] = valley_index_end2
        # else:
        #     valley_index_end = valley_index_end2

        # (k,1); [5,9,...,4,...,6,...]
        valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end), 
                                       (length) * torch.ones_like(valley_index_end), valley_index_end)
        
        # logging.info(f'valley_index_start:{valley_index_start}, valley_index_end:{valley_index_end}')
        
        _,counts = torch.unique(batch_index, return_counts = True) # (#batch, 1); the number of valleys in each sample
        max_counts = (torch.max(counts)).item() 

        utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(xs_pad.device)
        batch_index_mask = utri_mat1[counts]
        batch_index_mask = batch_index_mask.reshape(-1,1)
        batch_index_mask = batch_index_mask.nonzero()[:, 0]

        valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        # logging.info(str(valleys))
        
        # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        utri_mat = torch.tril(torch.ones(length+1,length),-1).to(xs_pad.device)
        output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        output_mask = output_mask.detach()

        # Aggregation
        alpha_h = torch.mul(uma_weights, xs_pad)
        xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, uma_weights).clamp_(1e-6)

        xs_pad = self.after_norm(xs_pad)
        # olens = (olens / olens[0] * xs_pad.shape[1]).type_as(olens)
        olens = counts
        # logging.info(f'olens:{olens}')
        # return xs_pad, olens, (uma_weights, p_attn)
        
        # if chunk_counts is None:
        #     logging.info(f'chunk_counts is None!')
        
        # return xs_pad, olens, chunk_counts
        return xs_pad, olens, (chunk_counts, (uma_weights, (valley_index_start, valley_index_end)))


        # for 
