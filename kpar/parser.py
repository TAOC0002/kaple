import os
def parse(parser):
    # Parse and set parameters
    parser.add_argument("--save_dir", default=None, type=str, required=True,
                        help="The save dir. Embeddings and indexes will be saved under this directory.")
    parser.add_argument("--function", default=None, type=str, required=True,
                        help="Specify the operation of k-par. Either 'construct_db' or 'query'")
    parser.add_argument("--patent_model_ckpt", default=None, type=str, required=True,
                        help="Path to the patent model checkpoint.")
    parser.add_argument("--pretrained_model_ckpt", default=None, type=str, required=True,
                        help="Path to the pretrained model checkpoint.")
    parser.add_argument('--meta_fac_adaptermodel', default='',type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_et_adaptermodel', default='',type=str, help='the pretrained entity typing adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default='', type=str, help='the pretrained linguistic adapter model')
    parser.add_argument("--corpus_index_file", default=None, type=str,
                        help="Path to the corpus index file (which is in .pkl format).")
    parser.add_argument("--corpus_content_file", default=None, type=str,
                        help="Path to the corpus content file (which is in .npy format).")
    parser.add_argument("--query_file", default=None, type=str,
                        help="Path to the query file (which is in .jsonl format).")
    parser.add_argument("--freeze_adapter", default=False, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="freeze the parameters of the roberta-large.")
    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,22", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=3, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_save_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--a_rate', type=float, default=0.5,
                        help="Rate of pre-trained LM loss")
    parser.add_argument('--b_rate', type=float, default=0.5,
                        help="Rate of adapter loss")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--normalize', type=bool, default=True,
                        help="Whether to apply normalization before cosine similarity measurement in the bi-encoder setting")
    parser.add_argument('--fusion_mode', type=str, default='concat',help='the fusion mode for bert feautre (and adapter feature) |add|concat|attentiom')
    # parser.add_argument('--sim_measure', type=str, default='cosine',
    #                     help="Similarity measure, used only when mode = 'bi'. Input one of (cosine, linear_transform)")
    
    args = parser.parse_args()
    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]

    return args