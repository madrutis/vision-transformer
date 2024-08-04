# Vision Transformer from Scratch
Implementation of the original [Vision Transformer](https://arxiv.org/pdf/2010.11929) paper for the [Michigan Data Science Team](https://mdst.club/)

**General goals of this project**
- Gain exposure to complex ML concepts
- Understand how transformers work and how they are implemented
- Gain experience implementing directly from an academic paper (good skill)
- Learn how to pair program with liveshare, git and improve python skills
- Get the chance to train a model with the cluster
- Be challenged, have fun, and meet some cool people!
- Possible applying and furthering an implementation or deploying to the cloud ---> Reach Goal


## Sample Project Timeline (Not Finalized)
### Week 1 (intros and setup)

- Project overview, intros, icebreakers
- Intro to Transformers
- Common Transformer use cases
- Set up ubuntu/zsh/github/liveshare/pip
- Gauge interest within transformers
- Split into individual groups (do a game for teambuilding or something)

### Week 2 (Linear Projection)
- Present on how images are patched and projected
- What is the purpose of the class token and positional encoding
- Architecture diagram
- Quick demo of the [einops](https://github.com/arogozhnikov/einops/tree/master) library
- Implementation / Q&A Time

### Week 3 (Attention pt 1)
- Present on how Attention works (video may be useful here too)
- Architecture Overview
- Implementation Time

### Week 4 (Attention pt 2 + Encoder)
- Present on how the encoder works, and uses the Attention block
- Architecture overview and taking a further look at paper 
- Implementation Time for attention and encoder

### Week 5 (MLP Head + Wrapping up implementation)
- Discussion on MLP Head for classification and training sets 
- Putting all of the pieces together and extracting class info from encoded images

### Week 6 (Wrapping Up implementation + training)
- Continue training/finetuning
- May split different groups up depending on certain datasets or hyperparameters depending on compute power


### Week 7 (Training/deployment/frontend)


### Week 8 (Presentation preparation / last minute improvements)



## Training Goals (TBD)
- May be possible to fully train a model on a smaller dataset, or finetune a preexisting model with for a more specialized task
- Once I try and train this model, I will have a better understanding of how this may look

## Deployment Goals (TBD)
- Look into deploying the model to an EC2 Instance with GPU
- Making a quick react frontend to query the model on EC2 for classicatikon

## Other Questions
- What tasks other than classification should we look into, given time constraints for the project