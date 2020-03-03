# Topic Models for Webpages Nested in Websites

## TopicModelHealthWebsite
Package for latent Dirichlet allocation (LDA) model with local topics (LT) and hierarchical asymmetric (HA) prior on document-topic distributions.

#### Model
We model a collection of websites and treat each webpage as a single document. Webpages are nested within websites.
The model extends LDA by directly modeling a single local topic for each website that is not shared by other websites. It further adds a hierarchical asymmetric prior on document-topic distributions. 

#### Usage
Minimally, only K, datapath, and savepath is required.
```
TopicModelHealthWebsites.HALT_LDA(K = K, datapath = "/data/dataanddict.jld", savepath = "/data/test.jld")
```
- K = number of global topics
- datapath = path to file containing the processed text data and dictionary
- savepath = path to save Gibbs sampling output
	
## Data
We provide data from webpages nested within 20 health department websites. 
