from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


# load pre-trained word embeddings model
print("Initializing... ")
# 400k-word GloVe model from https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html
glove_input_file = "neighbors/static/glove.6B.50d.txt"
model = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)

# create game window
global fig, ax
fig, ax = plt.subplots(1, 1, figsize=(7,7))
fig.subplots_adjust(right=0.85, top=0.85)
fig.canvas.manager.set_window_title('Word Associations')
# center axes, pre-animation
ax.set_xlim(left=-1, right=1)
ax.set_ylim(top=1, bottom=-1)

# initialize 'guess temperature' variables
colors = [[1,0.2,0.2],[0.9,0.2,0.2],[0.3,0.1,0.1],[0.1,0.3,0.1],[0.2,0.8,0.3]]
cmap = LinearSegmentedColormap.from_list("", colors)
temperature = 0.5
turnCount = 0

def selectWordPath(model):
    '''Semi-randomly selects initial and goal words for word associations game
    Word2Vec model -> (root word, goal word)'''
    root = np.random.choice(model.index_to_key[2000:5000]) 
    target = root
    targetList = [root]
    # make several passes to select goal word
    for i in range(3):
        result = model.most_similar(positive=[target],topn=5)
        wordList = np.array(result)[:,0] # selects column of just word strings
        # loops when random word chosen matches a previous target word
        while target in targetList:
            target = np.random.choice(wordList)
        # eliminate certain options from next turn's selections
        if i==0:
            for word in result:
                targetList.append(word[0])
        else:
            targetList.append(target)

    return (root, target)


def display_wordpca(model, root, goal):
    '''Displays PCA of given word and associated words
    Works interactively based on player clicks
    Word2Vec model, root word, goal word -> none'''
    
    global temperature, alpha
    alpha = 0
    # store axes of previous graph
    prevxLim = ax.get_xlim()
    prevyLim = ax.get_ylim()

    # store associated words data
    result = model.most_similar(positive=[root],topn=5)
    wordList = np.array(result)[:,0] 
    wordList = np.insert(wordList, 0, [root])        
    word_vectors = np.array([model[w] for w in wordList])

    # use PCA to two-dimensionalize vectors
    pca_model = PCA(n_components=2)
    twodim = pca_model.fit_transform(word_vectors)
    twodim = twodim-twodim[0] # center the original word


    ### PLOT ###
    
    # clear plot and remove axes labels
    ax.cla()
    ax.axes.axis('off')

    # plot point for target word
    target = ax.scatter(0, 0, s=90, c=[temperature], cmap=cmap, vmin=0, vmax=1, zorder=3, alpha=1) 
    
    if root != goal:
        #plot points for associated words
        coll = ax.scatter(twodim[:,0], twodim[:,1], s=40, c="#bbbbbb", zorder=2, picker=5, alpha=alpha)
        # label words
        labels = [ax.text(x+0.04, y+0.06, word, zorder=4, alpha=alpha) for word, (x,y) in zip(wordList, twodim)]
        # draw connecting lines
        lines = [ax.plot([0,x],[0,y], c="#eeeeee", linewidth=2.5, zorder=1, alpha=alpha) for word, (x,y) in zip(wordList, twodim)]
        lines = [lines[i][0] for i in range(len(lines))]
    

    ### ANIMATE ###
        
    # create 'frames' to animate the shift of axes from previous to current graph
    newxLim = ax.get_xlim()
    newyLim = ax.get_ylim()
    ax.set_xlim(prevxLim)
    ax.set_ylim(prevyLim)

    frameNum = 25
    xLeft = np.geomspace(prevxLim[0], newxLim[0], frameNum) # use logarithmic curve for smooth transition
    xRight = np.geomspace(prevxLim[1], newxLim[1], frameNum)
    yBottom = np.geomspace(prevyLim[0], newyLim[0], frameNum)
    yTop = np.geomspace(prevyLim[1], newyLim[1], frameNum)

    # indicate closeness of new guess with color temperature value
    howSimilar = model.similarity(root, goal)
    colorTransition = np.linspace(temperature, howSimilar, frameNum)
    temperature = howSimilar

    def transition(frame):
        ax.set_xlim(left=xLeft[frame], right=xRight[frame])
        ax.set_ylim(top=yTop[frame], bottom=yBottom[frame])
        target.set_array([colorTransition[frame]])

    animMove = FuncAnimation(fig, transition, frames=frameNum, interval=10, repeat=False)
    plt.pause(0.5)

    # end game if player selects goal word
    if root == goal:

        def greenify(frame):
            circleSize = target.get_sizes()[0]
            circleTransparency = target.get_alpha()
            newSize = circleSize*1.25
            target.set_alpha(circleTransparency*0.85)
            target.set_sizes([newSize])

        animGrow = FuncAnimation(fig, greenify, frames=40, interval=10, repeat=False)
        plt.pause(1.5)
        print(f"Congratulations! You found the right word in {turnCount} turns.")
        return
    
    
    ### MORE ANIMATIONS ###

    def onclick(event):
        '''Defines game action if player clicks a point.
        Increments turnCount, fades out graph, 
        calls function to re-graph with new root word'''
        fig.canvas.mpl_disconnect(cid)
        global turnCount
        turnCount += 1
        animOut = FuncAnimation(fig, fadeout, frames=10, interval=10, blit=False, repeat=False)
        plt.pause(0.2) # run animation before drawing new graph
        display_wordpca(model, wordList[event.ind][0], goal)
        return

    def fadeout(t):
        '''Fades out graph elements'''
        global alpha
        if alpha > 0:
            alpha -= 10
        coll.set_alpha(alpha/100)
        for line in lines:
            line.set_alpha(alpha/100)
        for label in labels:
            label.set_alpha(alpha/100)
        return
    
    def fadein(t):
        '''Fades in graph elements'''
        global alpha
        if alpha < 100:
            alpha += 10
        coll.set_alpha(alpha/100)
        for line in lines:
            line.set_alpha(alpha/100)
        for label in labels:
            label.set_alpha(alpha/100)
        return


    ### SHOW VISUALIZATION ###

    animIn = FuncAnimation(fig, fadein, frames=10, interval=10, blit=False, repeat=False)
    plt.pause(0.5)

    cid = fig.canvas.mpl_connect('pick_event', onclick) # enambles click functionality
    plt.show()


print("\nWORD ASSOCIATIONS\n")

(root, goal) = selectWordPath(model)

print(f"Your starting word is '{root.upper()}.' Navigate to '{goal.upper()}' in as few turns as possible!")

display_wordpca(model, root, goal)