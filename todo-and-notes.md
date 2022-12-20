- ~~Pass the action in as an embedding~~
  - Learn new state embedding so that the embedding can change to adapt with the added position embedding
  - ~~Understand why recent ViT models don't use the position encoding~~ (they just use conv for position)
  - ~~Try summing it with the state just like the position embedding~~
- ~~since loss is not decreasing, we should try to overfit on a single batch~~
- ~~visualize actual states and predicted states. right now we are not searching through a tree, so we should just be playing random actions and predicting the latent state results.~~
- ~~integrate transformer to take actions~~
- Resume accurate 1 step model, predict most likely state-action trajectories, and visualize them
- ~~See if multistep prediction problem is due to forwarding outside of training~~
- Count state visits
- ~~Reverse entropy to plan along well understood paths~~
- So I accidentally trained GPT from a randomized DVQ and was able to achieve high prediction accuracy. This suggests we can train them together or not train the DVQ at all to reconstruct...
- Try predicting action-states with sequences of image-patch-tokens, instead of or in addition to image-tokens as this should allow for better generalization / translation invariance. This as it enables the position encoding to learn 2D relationships vs now where the position encoding just represents a single dimensional frame number. An alternative easier solution would be to add a fully connected layer to the top of the DVQ before the quantization that learns 2D => 1D that can reconstruct. This hopefully will help build a 1D representation that conveys some translation invariance and other generalization which the transformer can use. Basically, spatial abstraction is already achievable in CNNs (e.g. classification). Pixel / feature prediction and self-supervised learning could also be helpful (or any supervised task that you have labels for) for creating a better spatial abstraction which can then be tokenized/discretized and used in GPT for temporal abstraction. Discretization seems to be critical for temporal abstraction in that it makes the number of predicted possibilities tractable for prediction in search trees.
- Rather than weight entropy by saliency level to encourage more abstract plans, perhaps the weighting should just be based on the total entropy expected divided by the expected duration of the abstract plan cf [Semi-MDP](http://user.engineering.uiowa.edu/~dbricker/Stacks_pdf1/SMDP_Intro.pdf).
- We can predict latent states for a single action, go-right, within a couple hours of training. Also, we have many aliased states (see [50-neighbor-knn](results/first_100_clusters_50_knn.PNG)). Finally we have many action aliases, e.g. right-up and right, noop and up, etc... So the 70k softmax at the end of GPT is 50 to 100x bigger than we need it.
- We should add extra states to the transformer softmax by using kmeans to see when we can increase k while maintaining a good centroid separation. Avg distance to centroid will decrease with larger k, but distance between centroids will usually decrease. We want to find times when this decrease is minimal, suggesting the new cluster is well separated from the others. The experiences that go into the above clustering should be obtained both externally and internally. By internally, I mean through some generative process (i.e. GAN, diffusion model) that produces novel token representations (e.g. humans living on mars, or a huge snake with wings) and attempts to integrate these representations into a plan. It's unclear how to best find which generated tokens will lead to the most learning, but it should be driven by things like, "humans on mars will avoid extincition and therefore provide many more opportunities for learning in the long-term" or "huge snakes with wings would be cool/scary and drawing them would allow sharing the idea with others hence increasing learning in a group of people"
- Language
- Use pin_memory in dataloader when using disk-backed replay buffer
- Allow evaluating hypotheticals within externally unvisited states. Most likely this happens by creating states internally and simulating several of those. Also, we should be able to tell when a low level plan, like reading a book, does not affect a high level plan, like taking a ship to Mars. In this case, the high level context (riding to Mars) should not change the prediction at a low level that reading a book is unlikely to physically change much around us. We don't even need to plan and simulate the result of reading a book while riding to Mars. The upper level context simply doesn't carry enough weight with regard to the lower level that we need to consider it affecting the lower level. I.e. it doesn't matter if I read the book at home or on the way to Mars, the outcome is basically the same. However, if I hammer a nail into the wall, it will have a drastically different effect at home vs on a ship. This type of reasoning is not yet handled by the learnmax model that I can think of. It should be able to forward simulate that puncturing the ship is much more dangerous than puncturing a wall on earth. I suppose a language model could try to query these types of plans and relate them to the sensorimotor model... Granted puncturing any vehicle is unusual. The context could also be like that of a language model, in that it just comes from a previous token instead of from some higher level model. So internally you'd have the state for
 riding a ship to mars, followed by reading a book, and then subsequent tokens would signal that this is safe due to internal simulations from reading a book in different contexts and integration with a language model that can glean this from others' experience. The salient token can be an average of the concrete tokens below that make it up. There should be a higher weighting of tokens at the end and beginning in this average as it's possible that there are many inconsequential tokens (e.g. waiting, meandering) in the middle. Then this new salient token can be added as training data to prompt context for the lower level.
- Perhaps have a patch saliency level below image that allows reasoning about parts of the sensor input. It seems the bottom two levels are special, one is 2D patch based, and two is image based and sits at the same level as actions. (vq-vae2 has something similar)
- Don't pass the flattened image to GPT. Just use the integer indexes. Then use KNN to reduce the number of tokens from 4096. Then when encountering a new cluster (judged by distance from old cluster and recon error), update the dvq so that the new cluster (say with the key gone) is correctly identified and mapped to a lower cardinality token space with KNN. We can also significantly reduce the token size at that point. Alternatively, we can just look at the effect of a reduced sequence window length.
- Reduce the 8192 clusters with pixel distance
- New clusters are not good enough to be used on their own. But I think adding a reduced (~700) cluster embedding to the cull 8192 cluster embedding will help the transformer a lot. Then we can try removing the z_q embedding which will allow us to change the embedding size. And more importantly let's try reducing the sequence window size and the action space size. The action space should maybe also be encoded as an addition of LEFT, RIGHT, FIRE (jump), UP (ladders), and DOWN where NOOP is zero or another embedding. The idea is to simplify things to the point where we can train zuma quickly and get to the salient event work. Then if that works, we can scale quantization to other tasks with a working baseline task.
- Try small sequence window (1, 2, 3...)
- Try resets after small number of actions (5 or so) enough to fall off platform, still get good prediction accuracy, and get to salient states
- Try states and actions in different heads - this means you don't have entropy associated with a given action over states in one forward pass. You need a batch that tries every action and looks at the state head to compare entropies across states. This means in tree_search you need to do a batch with all actions for each branch and tree search is already the bottleneck going just fast enough to act real-time. It also complicates things quite a bit and requires a lot of change. It does reduce the search space per forward pass by 20x for states and 1000x for actions and gets more predictive power out of the same network size. The sequence window would be more limited though as now you have action and state tokens.
- Add a fully connected layer to the top of the encoder so that it learns the flatten transformation and passes 2D info through activations instead of implicitly by unflattening in the decoder. According to Deepmind's flamingo paper, position info gets encoded into the channels and therefore also in the 1D flattening (pooling is mentioned in referenced paper but I think they are confounding that with flattening) - although from the code it looks like they use max pooling, so it may be worth trying if 2D position embeddings do help
 > These visual features are obtained by first adding a learnt temporal position encoding to each spatial grid of features corresponding to a given frame of the video (an image being considered as a single-frame video). Note that we only use temporal encodings and no spatial grid position encodings; we did not observe improvements from the latter, potentially because CNNs implicitly encode space information channel-wise (Islam et al., 2021).
- Thinking more about 
- Action reduction for zuma
    0 Action.NOOP
    1 Action.FIRE - jump
    2 Action.UP - move up ladder
    3 Action.RIGHT 
    4 Action.LEFT
    5 Action.DOWN
    6 Action.UPRIGHT - move up right ladder not needed
    7 Action.UPLEFT - move up left ladder not needed
    8 Action.DOWNRIGHT - move down right ladder not needed
    9 Action.DOWNLEFT - move up left ladder not needed
    10 Action.UPFIRE - jump up ladder not needed
    11 Action.RIGHTFIRE - jump right ladder not needed
    12 Action.LEFTFIRE - jump left ladder not needed
    13 Action.DOWNFIRE - jump down ladder not needed
    14 Action.UPRIGHTFIRE - jump up right ladder not needed
    15 Action.UPLEFTFIRE - jump up left ladder not needed
    16 Action.DOWNRIGHTFIRE - jump down right ladder not needed
    17 Action.DOWNLEFTFIRE - jump down left ladder not needed
- Create a dummy transition model of ints => ints for each action that allows modeling without Atari. They can just be deterministic random transitions, e.g. state 0 + action 0 => state 420, state 0 + action 1 => state 69, etc...  We can then create arbitrarily small transition tables and see how model scales. This is similar to doing the short reset thing though. It seems to be just picking the same action it's seen before (judging from lack of perf improvement when reducing actions)
- Salient loss - we can predict the sum of the next n-frames (summing all patches or all single tokens) and compute the MSE or softmax/cross-entropy for loss. This ensures that the logits or last layer activations contain information necessary for determining when a large change in possibilities happens. A problem, however occurs when we start following the same salient path over and over, in which case the possibilities change will be more gradual. In this case we are forgetting the other possibilities before and after the salient state, and hardcoding to the possibilities that occur in the salient path. I guess this is okay, so long as we don't recalculate salient states and forget that the state is salient. Even if we do, the lower level will just follow the salient path and forget that it's possible to not follow the salient path in which case it sort of acts like salient compression.
- Make a legit dataset that we can do real epics with. Start with small max episode steps and make sure test set does not contain sequences in the train set. This will require some hash stable of sequences in train. Then we can start to see how hard it is to get high test prediction accuracy with low environment complexity (i.e. small transition table due to small episode size). We can also train on this with traditional dataloaders etc... as we don't need to run Atari.
- Patch based DVQ
  - Concat patches along sequence window with delimiter token
  - GPT accuracy support
  - Actions should be given one part of the sequence window, instead of added to every token in the sequence. You could add to every patch token I guess, but that forces the whole network to disentangle the action from the patch whereas many times the next frame patch at the same position does not depend on the action (like the background or skull)
  - Due to above, we don't have action-states in each token. This is nice as we output an action. The disadvantage is that the sequence window size grows a lot. We reduced the embedding size to ~70(emb_d)*441(n_patch)=30870+action from 4410 for a net growth in input size of 10x therefore (n^2) 100x net size! So we'll have to reduce the sequence window to less than 10 likely.
    To extend the window without N**2 more weights, we could use https://arxiv.org/abs/2203.08913 or referenced methods. We can try reducing the number of patches though from 441 to 144 = 12*12 as atari img's are 84x84 and therefore patch sizes would be 7x7 instead of 4x4 pixels.
    Okay so we reduced to 11x11 patches = 121 from 441. With emb_d = 70, we have 121*70 = 8470 length of image tensor vs 4410 from before. 
    Thing is we can just change the embedding length now separate from what DVQ outputs. So n_emb is the really important thing.
    Would be good to know that this embedding size can be used for reconstruction though. Let's try n_emb of 128 and strides=4,4, so conv filter will be ~6 (5.25) so 36 token length. 36*25=900
    What works is stride 2x4 as 8 goes into 84 => 11x11 = 121 patches, so 121*30=3630
    
  - From Gato: 
    - Use an observation separator to distinguish between frames? DONE
    - Also add an in-image position encoding to the patch (should not be needed per Flamingo?). DONE - 1D
    - Tokenize images into 16x16, normalize -1=>1 and divide by root(16) - Shouldn't need as we learn a nice discrete embedding
    - Encode image patches through one resnet block - Shouldn't need as we learn a nice discrete embedding
    - They're narrower transformers 2048, 1536, 768 than me at around 4k and much more compute. Their patch embeddings at 768 must be small, like 10??

- Find out why the learned embedding (derived from z_q_ind ints) is so much better than the z_q_emb embedding from the top of the auto-encoder. We really need to be able to use semantic embeddings so that similar experiences can be used for task transfer.
- Detect salience on a pre-recorded dataset of Montezuma so that we don't have to learn to play and can focus on salience detection.
- See if we can use the methods discussed in Progress and Compress https://arxiv.org/abs/1805.06370
- In paper, note how we address requirements of [_Biological underpinnings for lifelong learning machines_](https://www.nature.com/articles/s42256-022-00452-0)
- Use single token in parallel with patch tokens by prefixing sequences with single token representation. Try projecting the 4410 dim embedding to 30 as well as training a 30 dim dvq for this. Such an "anchor" representation is critical for generalization across time and helps efficiency by allowing transfer learning across salience levels. Should be able to get key without this, so we can delay implementation.
- Train on level 1 salience and produce level 2 salience, visualize
Notes:
#### Comparison with hierarchy of abstract machines (HAMs)
These have discrete action, call, choice, and stop where execution is occurring at one level of abstraction at a time (i.e. sequentially). A big difference between this and learnmax is that learnmax can search within different levels of abstraction in parallel. Since high level plans don't change as often, most searching is done in the lower levels even when executing a high level plan to find some long term entropy. So your basically optimizing for entropy reduction per unit time. However since high level entropy possibly unlocks new vistas and worlds of low level entropy, we still should perhaps afford more weight to high level entropy just based on level alone. 

#### Comparison with options in hierarchical RL
It seems that options may support planning at multiple levels simultaneously. However, I don't see a way to automatically learn the options. Rather they are provided by humans. 


#### Comparison with human visual cortex
Number of possible images too large to represent with ints (single token). Similarly for visual cortex, third layer is made up of combination of discrete 
representations (lines in different directions) from layer 2. i.e. there is no single grandmother neuron. https://en.wikipedia.org/wiki/Grandmother_cell


#### Comparison with "From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning"
The set of possible salient states above you which you are looking for is similar to the "grounding set" mentioned in Definition 1. However, 
there's no need to specify states derived from set operations on these states as such combinations should be learned by the transformer.

