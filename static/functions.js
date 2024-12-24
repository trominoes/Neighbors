

// INITIALIZATION //


/* populateGrid() is called on load of the neighbors page.
   It populates the hexGrid with 163 li.hex elements, which 
   corresponds to 13 rows of 13-12-13-12... alternating 
   length. Each li has a child p.hexText element.
*/
function populateGrid() {
    const ul = document.getElementById('hexGrid');
    for (let i = 0; i < 163; i++) {
        const li = document.createElement('li');
        li.classList.add("hex");
        const p = document.createElement('p');
        p.classList.add("hexText");
        li.appendChild(p);
        ul.appendChild(li);
    }
}

/* fetchWords() is called on load of the neighbors page.
   It fetches the JSON list of starting words and JSON 
   dictionary of neighbors, and populates the hex structure
   with a selected start word and its neighbors.
*/
async function fetchWords() {
    try {
        // fetch both endpoints concurrently
        const [response1, response2] = await Promise.all([
            fetch('static/wordlist.json'),
            fetch('static/neighbordict.json')
        ]);

        // wait for both responses to be parsed as JSON
        const wordList = await response1.json();
        const neighborDict = await response2.json();
        window.neighborDict = neighborDict;

        // select start and destination words
        const start = selectStart(wordList);
        const trackerStart = document.getElementById("solve-start");
        trackerStart.textContent = start;
        const modalStart = document.getElementById("howto-start");
        modalStart.textContent = start;
        window.start = start;

        const dest = selectDestination(start);
        const trackerDest = document.getElementById("solve-dest");
        trackerDest.textContent = dest;
        const modalDest = document.getElementById("howto-dest");
        modalDest.textContent = dest;
        window.dest = dest;

        // set global variables
        window.winloss = false;
        window.reveal = false;

        // add start word, neighbors to hex structure
        initializeHexes();

        // fade in welcome screen
        const welcomeElements = document.querySelectorAll(".welcome-elements")[0];
        welcomeElements.style.opacity = 1;

        const welcomeDate = document.querySelectorAll(".welcome-date")[0];
        const today = new Date();
        const formattedDate = today.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        welcomeDate.textContent = formattedDate;

    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

/* initializeHexes() generates the initial grid
   layout, given the window.start word. 
*/
function initializeHexes() {
    // initialize step counter
    window.steps = 0;
    // set root/neighbors to game's starting position
    const hexCells = document.getElementsByClassName("hex");
    const initialRoot = hexCells[81];
    initialRoot.id = "hex-root";
    markNewNeighbors(initialRoot);

    // add start word and neighbors to hex structure
    initialRoot.children[0].textContent = window.start;

    const neighbors = document.querySelectorAll(".neighbors");

    for (let i = 0; i < neighbors.length; i++) {
        let cell = neighbors[i];
        cell.children[0].textContent = window.neighborDict[window.start][i];
    }

    // center and fade in
    centerGridAroundRoot();
    const hexContainer = document.querySelector('#hex-container');
    setTimeout(function () {
        hexContainer.style.opacity = "100%";
    }, 300);

}


// NAVIGATION & OVERLAYS //


function redirectNeighbors() {
    window.location.href = "/neighbors";
}

function redirectMethods() {
    window.location.href = "/neighbors/methodology";
}

function openInfoModal() {
    const modal = document.getElementById("modal-info");
    modal.style.display = "block";
    document.body.style.overflow = 'hidden';
}

function closeInfoModal() {
    const modal = document.getElementById("modal-info");
    modal.style.display = "none";
    document.body.style.overflow = '';
}

function showWin() {
    const overlay = document.getElementById("win-loss-overlay");
    const heading = document.getElementById("win-loss-heading");
    const desc = document.getElementById("win-loss-desc");
    heading.textContent = "Thanks for playing!";
    desc.textContent = "You completed today's puzzle in " + window.steps.toString() + " moves."
    overlay.style.backgroundColor = "#c5ccbd";
    showWinLoss();
}

function showLoss() {
    const overlay = document.getElementById("win-loss-overlay");
    const heading = document.getElementById("win-loss-heading");
    const desc = document.getElementById("win-loss-desc");
    heading.textContent = "Nice try!";
    desc.textContent = "You couldn't reach the destination word in six moves. See a solution, or try again..."
    overlay.style.backgroundColor = "#ddd";
    showWinLoss();
}

function showWinLoss() {
    const grid = document.getElementById("hexGrid");
    const container = document.getElementById("hex-container");
    const overlay = document.getElementById("win-loss-overlay");

    grid.style.opacity = 0;
    overlay.classList.remove("hidden");
    container.style.maskImage = "none";
}

function closeWinLoss() {
    const grid = document.getElementById("hexGrid");
    const container = document.getElementById("hex-container");
    const overlay = document.getElementById("win-loss-overlay");

    grid.classList.remove("locked");
    overlay.classList.add("hidden");
    setTimeout(function () {
        grid.style.opacity = 1;
        container.style.maskImage = "radial-gradient(circle, rgba(255, 255, 255, 1) 40%, rgba(255, 255, 255, 0) 100%)";
    }, 100);
}


// GAME FUNCTIONS //


/* selectStart(words) randomly chooses & returns one start word
   from the list of words passed as an argument. Utilizes
   the createHashIndex or randomizeIndex functions for 
   this random selection.
*/
function selectStart(words) {
    const index = createHashIndex(words.length);
    // replace above with randomizeIndex to change word selection
    return words[index];
}

/* createHashIndex(length) returns a random date-based index 
   between 0 and length. One word is selected per day.
*/
function createHashIndex(length) {
    const referenceDate = new Date('2024-12-13');
    const currentDate = new Date();

    const timeDifference = currentDate - referenceDate;
    const daysDifference = Math.floor(timeDifference / (1000 * 60 * 60 * 24));

    // map days difference to an index between 0 and length
    const index = daysDifference % length;

    return index;
}

/* randomizeIndex(length) returns a random index between 
   0 and length. A new word is selected every function call.
*/
function randomizeIndex(length) {
    const randomFactor = Math.random();
    return Math.floor(randomFactor * length);  // round down to an integer index
}

/* selectDestination(root) randomly chooses & returns a destination 
   word at least one and at most three steps away from the start 
   word, using a browser-replicable seeded random number generator 
   based on the cyrb128 hash and sfc32 PRNG.

   The solution path is stored in the window.solution variable.
*/
function selectDestination(root) {
    let target = root;
    let solution = [root];
    let blocklist = [root];

    const today = new Date();
    const hashSeed = cyrb128(today.toDateString());
    let random = sfc32(hashSeed[0], hashSeed[1], hashSeed[2], hashSeed[3]);

    // first pass of selecting start word's neighbor
    let neighbors = window.neighborDict[target];
    let randIndex = Math.floor(random() * 6);
    target = neighbors[randIndex];
    solution.push(target);
    // add start word's immediate neighbors to blocklist
    for (let index = 0; index < 6; index++) {
        blocklist.push(neighbors[index]);
    }

    // continue selecting target word with depth 3
    for (let i = 0; i < 2; i++) {
        // disallow target from being in blocklist
        neighbors = window.neighborDict[target];
        while (blocklist.includes(target)) {
            randIndex = Math.floor(random() * 5);
            target = neighbors[randIndex];
        }
        blocklist.push(target);
        solution.push(target);
    }

    window.solution = solution;
    return target;
}

/* Helper function for selectDestination: creates a 128-bit random seed
   given any input string. From
   https://stackoverflow.com/questions/521295/seeding-the-random-number-generator-in-javascript.
*/
function cyrb128(str) {
    let h1 = 1779033703, h2 = 3144134277,
        h3 = 1013904242, h4 = 2773480762;
    for (let i = 0, k; i < str.length; i++) {
        k = str.charCodeAt(i);
        h1 = h2 ^ Math.imul(h1 ^ k, 597399067);
        h2 = h3 ^ Math.imul(h2 ^ k, 2869860233);
        h3 = h4 ^ Math.imul(h3 ^ k, 951274213);
        h4 = h1 ^ Math.imul(h4 ^ k, 2716044179);
    }
    h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067);
    h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233);
    h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213);
    h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179);
    h1 ^= (h2 ^ h3 ^ h4), h2 ^= h1, h3 ^= h1, h4 ^= h1;
    return [h1 >>> 0, h2 >>> 0, h3 >>> 0, h4 >>> 0];
}

/* Helper function for selectDestination: generates random number 
   between 0 and 1 based on four 32-bit seeds. From
   https://stackoverflow.com/questions/521295/seeding-the-random-number-generator-in-javascript.
*/
function sfc32(a, b, c, d) {
    return function () {
        a |= 0; b |= 0; c |= 0; d |= 0;
        let t = (a + b | 0) + d | 0;
        d = d + 1 | 0;
        a = b ^ b >>> 9;
        b = c + (c << 3) | 0;
        c = (c << 21 | c >>> 11);
        c = c + t | 0;
        return (t >>> 0) / 4294967296;
    }
}

/* revealSolution() reveals the daily puzzle's computed
   solution for the player, after the player has won or 
   lost the game. Each step in the solution is animated.
*/
function revealSolution() {
    revealSteps(function() {
        // this code passed as a callback function to revealSteps; executes after revealSteps is completed
        const grid = document.getElementById("hexGrid");
        grid.classList.remove("locked");
    });
}

/* revealSteps() is a helper function to revealSolution,
   animating each step in the solution while the grid
   does not respond to user input.
*/
function revealSteps(callback) {
    resetPuzzle();

    // keep hex grid locked while revealing solution
    const grid = document.getElementById("hexGrid");
    grid.classList.add("locked");
    window.reveal = true;

    // animate solution steps
    let j = 1;
    const interval = 1000;

    function showNextStep() {
        if (j < window.solution.length) {
            let step = window.solution[j];
            // find neighboring word with next step of solution
            let target;
            const hexCells = document.getElementsByClassName("neighbors");
            for (let i = 0; i < hexCells.length; i++) {
                if (hexCells[i].children[0].textContent == step) {
                    target = hexCells[i];
                }
            }
            // "click" the neighboring word and repeat
            onClickNeighbor(target);
            j++;
            setTimeout(showNextStep, interval);
        } else {
            callback();
        }
    }

    setTimeout(() => {
        showNextStep();
    }, 1000);
}

/* resetPuzzle() returns the hex grid to its original
   setup, giving the player another opportunity to solve
   the puzzle.
*/
function resetPuzzle() {
    // reset global vars
    window.winloss = false;

    // reset step tracker
    const destTracker = document.getElementById("solve-dest");
    destTracker.classList.remove("found");
    const stepTracker = document.querySelectorAll(".solve-attempt");
    stepTracker.forEach(elem => {
        elem.remove();
    });
    const elems = document.querySelectorAll(".solve-elem");
    elems.forEach(elem => {
        elem.style.gridRowStart = 1;
    });

    // reset puzzle & close overlay
    const ul = document.getElementById('hexGrid');
    ul.innerHTML = '';
    populateGrid();
    initializeHexes();
    closeWinLoss();
}


// GRID OPERATIONS //


/* onClickNeighbor(target) is called when a .neighbor
   hex cell is clicked (passed as 'target'). It refreshes 
   the grid with new neighbor words and highlighted cells.
*/
function onClickNeighbor(target) {
    let neighbors = document.querySelectorAll(".neighbors");
    const grid = document.getElementById("hexGrid");
    const prevRoot = document.getElementById("hex-root");
    const destTracker = document.getElementById("solve-dest");
    const clickWord = target.children[0].textContent;

    // increment step trackers
    window.steps++;
    if (window.steps <= 6 && !window.winloss) {
        updateStepTracker(clickWord);
    }
    destTracker.classList.remove("found");

    // erase text from neighbors (if not previous root or target cells)
    neighbors.forEach(neighbor => {
        if (neighbor != target && neighbor != prevRoot) {
            neighbor.children[0].textContent = "";
        }
    });

    // unmark old neighbors, mark new neighbors
    unmarkNeighbors();
    markNewNeighbors(target);

    // populate new neighbors
    neighbors = document.querySelectorAll(".neighbors");

    let i = 0;
    neighbors.forEach(neighbor => {
        let newNeighbor = window.neighborDict[clickWord][i];
        if (newNeighbor == prevRoot.children[0].textContent) {
            i++;
            newNeighbor = window.neighborDict[clickWord][i];
        }
        if (neighbor != prevRoot) {
            neighbor.children[0].textContent = newNeighbor;
            i++;
        }
    });

    // set ID of new root
    prevRoot.removeAttribute('id');
    target.id = "hex-root";

    // re-center grid
    centerGridAroundRoot();

    // show win/loss overlay if game complete
    if (!window.winloss) {
        if (clickWord == window.dest) {
            grid.classList.add("locked");
            destTracker.classList.add("found");
            window.winloss = true;
            if (!window.reveal) {
                setTimeout(() => {
                    showWin();
                }, 400);
            }
        } else if (window.steps == 6) {
            grid.classList.add("locked");
            window.winloss = true;
            setTimeout(() => {
                showLoss();
            }, 400);
        }
    }

}

/* updateStepTracker(word) adds the player's move
   to the step tracker, indicating that another 'word'
   has been visited. The destination/comparison arrow is shifted
   accordingly.
*/
function updateStepTracker(word) {
    const tracker = document.getElementById("step-tracker");

    // add player's move to the tracker
    const step = document.createElement("span");
    step.classList.add("underline", "solve-attempt");
    step.textContent = word;
    tracker.appendChild(step);

    // shift arrow, destination word down by a row
    const elems = document.querySelectorAll(".solve-elem");
    elems.forEach(elem => {
        elem.style.gridRowStart = window.steps + 1;
    });

    // animate new step
    step.style.transform = "scale(1.1)";
    setTimeout(() => {
        step.style.transform = 'scale(1)';
    }, 0);
}

/* unmarkNeighbors() visits all the li.hex elements 
   marked as neighbors and removes the class from them.
*/
function unmarkNeighbors() {
    const neighborCells = document.querySelectorAll(".neighbors");

    neighborCells.forEach(neighbor => {
        neighbor.classList.remove('neighbors');
    });
}

/* markNewNeighbors(target) visits all the neighboring
   cells around target and marks them as neighbors. Accounts
   for edges of the grid, i.e. only accesses cells in-bounds.
*/
function markNewNeighbors(target) {
    const hexCells = document.getElementsByClassName("hex");
    const newRoot = Array.from(hexCells).indexOf(target);

    hexCells[newRoot].classList.add('visited');

    if (newRoot % 25 == 12) { // odd row, end
        if (newRoot > 12) { // not first row
            hexCells[newRoot - 13].classList.add('seen', 'neighbors');
        }
        if (newRoot < 150) { // not last row
            hexCells[newRoot + 12].classList.add('seen', 'neighbors');
        }
        hexCells[newRoot - 1].classList.add('seen', 'neighbors');
    } else if (newRoot % 25 == 24) { // even row, end
        hexCells[newRoot - 13].classList.add('seen', 'neighbors');
        hexCells[newRoot - 12].classList.add('seen', 'neighbors');
        hexCells[newRoot + 12].classList.add('seen', 'neighbors');
        hexCells[newRoot + 13].classList.add('seen', 'neighbors');
        hexCells[newRoot - 1].classList.add('seen', 'neighbors');
    } else if (newRoot % 25 == 0) { // odd row, start
        if (newRoot > 12) { // not first row
            hexCells[newRoot - 12].classList.add('seen', 'neighbors');
        }
        if (newRoot < 150) { // not last row
            hexCells[newRoot + 13].classList.add('seen', 'neighbors');
        }
        hexCells[newRoot + 1].classList.add('seen', 'neighbors');
    } else if (newRoot % 25 == 13) { // even row, start
        hexCells[newRoot - 13].classList.add('seen', 'neighbors');
        hexCells[newRoot - 12].classList.add('seen', 'neighbors');
        hexCells[newRoot + 12].classList.add('seen', 'neighbors');
        hexCells[newRoot + 13].classList.add('seen', 'neighbors');
        hexCells[newRoot + 1].classList.add('seen', 'neighbors');
    } else {
        if (newRoot > 12) {
            hexCells[newRoot - 13].classList.add('seen', 'neighbors');
            hexCells[newRoot - 12].classList.add('seen', 'neighbors');
        }
        if (newRoot < 150) {
            hexCells[newRoot + 12].classList.add('seen', 'neighbors');
            hexCells[newRoot + 13].classList.add('seen', 'neighbors');
        }
        hexCells[newRoot - 1].classList.add('seen', 'neighbors');
        hexCells[newRoot + 1].classList.add('seen', 'neighbors');
    }
}

/* centerGridAroundRoot() computes and applies the grid margins
   necessary to center the root hex cell within the hex container
   element. The grid is parented by the hex container, but it 
   overflows, so its centered position is set by moving its margins
   according to the root hex cell's relative position.
*/
function centerGridAroundRoot() {
    const flexContainer = document.querySelector('#hex-container');
    const gridContainer = document.querySelector('#hexGrid');
    const targetItem = document.querySelector('#hex-root');

    // get the bounding rectangles of the flex container, target element
    const flexRect = flexContainer.getBoundingClientRect();
    const targetRect = targetItem.getBoundingClientRect();

    // calculate the center of the flex container
    const flexCenterX = flexRect.left + (flexRect.width / 2);
    const flexCenterY = flexRect.top + (flexRect.height / 2);

    // calculate the center of the target element, in relation to center of flex container
    const targetCenterX = (targetRect.left + (targetRect.width / 2)) - flexCenterX;
    const targetCenterY = (targetRect.top + (targetRect.height / 2)) - flexCenterY;

    // determine existing margins of the grid
    const gridMarginLeft = gridContainer.style.marginLeft ? parseInt(gridContainer.style.marginLeft, 10) : 0;
    const gridMarginTop = gridContainer.style.marginTop ? parseInt(gridContainer.style.marginTop, 10) : 0;

    // determine the margins needed to center the grid
    const marginLeft = gridMarginLeft - 2 * targetCenterX; // doubled because grid is centered?
    const marginTop = gridMarginTop - targetCenterY;

    // apply the calculated margins to the grid container
    gridContainer.style.marginLeft = `${marginLeft}px`;
    gridContainer.style.marginTop = `${marginTop}px`;
}