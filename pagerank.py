import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    
    


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_of_pages = len(corpus)
    transition_m = {}
    
    if len(corpus[page]) == 0:
        for page_c in corpus:
            transition_m[page_c] = round(1/num_of_pages, 4) 
        # Making sure it works as expected
        """
        try:
            sum_ = sums_dict(transition_m)
            assert  round(sum_ , 4)== 1
        except:
            print(sum_)
            print(transition_m)
            sys.exit()
        
        """
        return transition_m                                        
        
    else: 
        uniform_prob =  (1-damping_factor)/num_of_pages
        for page_c in corpus:
            transition_m[page_c] = round(uniform_prob, 4)
        linked_pages_set = corpus[page]
        num_linked_pages = len(linked_pages_set)
        for linked_page in corpus[page]:
            transition_m[linked_page] += round(damping_factor/num_linked_pages, 4)
        # TODO: Making sure it works as expected
        """
        try:
            sum_ = sums_dict(transition_m)
            assert  round(sum_ , 4)== 1
        except:
            print(sum_)
            print(transition_m)
            sys.exit()
        
        """
        
    return transition_m

def sums_dict(dict_):
    sum_ = 0
    for key in dict_:
        sum_ += dict_[key]
    return sum_
def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample_pr = {}
    previous_page = list(corpus.keys())[random.randint(0, len(corpus)-1)]
    for _ in range(n):
        # Get the transition model from previous page
        transition_m = transition_model(corpus, previous_page, damping_factor)
        next_page = bernoulli_trial(normalize_tm(transition_m))
        previous_page = next_page
        previous_count = sample_pr.get(next_page, 0)
        sample_pr[next_page] = previous_count + 1 
    for page in sample_pr:
        sample_pr[page] /= n
    return sample_pr

def normalize_tm(transition_m):
    """
    Receives a transition model (a dictionary), then returns two lists.
    One is the name of the page, and the other contains the cummulative
    probabilities to perform a Bernoulli trial.
    """
    prob = 0
    page_names, norm_probs = [], []
    for page, prob_value in transition_m.items():
        page_names.append(page)
        prob += prob_value
        norm_probs.append(round(prob, 4))
    assert len(page_names) == len(norm_probs), "Page names and Norm probs must be of the same size"
    try:
        assert norm_probs[-1] >= 0.99
    except:
        print("Last value in normalized probabilities less than expected",norm_probs[-1])
    return [page_names, norm_probs]

def bernoulli_trial(page_names_norm_probs):
    """
    Randomly chooses a page_name based on their corresponding
    norm_probs. I.e. the higher the original probalitiy value of
    a page from its transition model, the higher the chances it's selected.
    Actually, the probs of being selected are equal to the prob values from 
    the transition model.
    """
    # norm_probs example: [0.1, 0.45, 0.67, 0.99] Always in increasing order
    page_names = page_names_norm_probs[0]
    norm_probs = page_names_norm_probs[1]
    rand_float = round(random.uniform(0, 1),5)
    for index, norm_prob in enumerate(norm_probs):
        if norm_prob > rand_float:
            return page_names[index]
    return page_names[-1]

def generate_parent_nodes(corpus):
    """
    Goes through the corpus and creates a dictionary with pages as keys,
    and the pages that link to those pages(keys) as values.
    """
    parent_nodes = {}
    for page in corpus:
        linked_pages = corpus[page]
        if linked_pages:
            for linked_page in linked_pages:
                parents_set = parent_nodes.get(linked_page, set())
                parents_set.add(page)
                parent_nodes[linked_page] = parents_set
        else:
            # if the page does not link to any page, we act as though
            # it links to all the pages. (including itself)
            for page2 in corpus:
                parents_set = parent_nodes.get(page2, set())
                parents_set.add(page)
                parent_nodes[page2] = parents_set

    return parent_nodes

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    iterate_pr = {}
    parent_nodes = generate_parent_nodes(corpus)
    num_of_pages = len(corpus)
    uniform_prob = (1-damping_factor)/num_of_pages
    difference = 1
    
    for page in corpus:
        iterate_pr[page] = 1/num_of_pages
    while difference > 0.001:
        largest_pr_delta = -1 # start at negative one because the new_pr_delta will be absolute values ( > 0)
        for page in corpus:
            summation = 0
            for parent in parent_nodes[page]:
                if len(corpus[parent]) > 0:
                    summation += iterate_pr[parent]/len(corpus[parent])
            new_pr = uniform_prob + damping_factor * summation
            new_pr_delta = abs(iterate_pr[page] - new_pr)
            iterate_pr[page] = new_pr
            if new_pr_delta > largest_pr_delta: # We don't want to stop as long as the condition is not met.
                largest_pr_delta = new_pr_delta # so as soon as any pr change is greater than the largest difference,
                                    # we must update. All of the pr changes have to be
                                    # smaller than 0.001 (pr: Page Rank)  
        difference = largest_pr_delta 
            
    return iterate_pr


if __name__ == "__main__":
    main()
