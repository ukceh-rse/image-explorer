let img;
const API_URL = 'http://localhost:8000';

let images = [];
let urls=[];
let selectedImg;
let SIDEBAR_WIDTH;
let THUMBNAIL_SIZE;
const GRID_GAP = 10;

function preload() {
  // Will hold our image
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  SIDEBAR_WIDTH = windowWidth / 2;
  THUMBNAIL_SIZE = windowHeight / 4;
  loadRandomImage().then(() => {
    selectedImg = img;
  });
}

function formatDateTime(filename) {
    // Extract date part: SITE_20190615_120000_xxx.jpg -> 20190615_120000
    const datePart = filename.split('_').slice(1, 3).join('_');
    // Parse into Date object
    const year = datePart.slice(0, 4);
    const month = datePart.slice(4, 6);
    const day = datePart.slice(6, 8);
    const hour = datePart.slice(9, 11);
    const min = datePart.slice(11, 13);
    return `${year}-${month}-${day}\n${hour}:${min}`;
  }

function draw() {
  background(220);
  if (img) {
    // Center image
    const x = 0; // (width - img.width) / 2;
    const y = 0; // (height - img.height) / 2;
    image(img, x, y);
  }

   // Draw grid on right
   const cols = Math.floor(SIDEBAR_WIDTH / (THUMBNAIL_SIZE + GRID_GAP));
   const startX = windowWidth - SIDEBAR_WIDTH;
   

   images.forEach((img, i) => {
     const row = Math.floor(i / cols);
     const col = i % cols;
     const x = startX + col * (THUMBNAIL_SIZE + GRID_GAP);
     const y = row * (THUMBNAIL_SIZE + GRID_GAP);
     image(img, x, y, THUMBNAIL_SIZE, THUMBNAIL_SIZE);

     fill(255);
     textAlign(CENTER);
     textSize(12);
     text(formatDateTime(urls[i]), 
          x + THUMBNAIL_SIZE/2, 
          y + THUMBNAIL_SIZE - 30);
   });

}

async function loadRandomImage() {
  try {
    const response = await fetch(`${API_URL}/random`);
    const data = await response.json();
    const imageUrl = data.url; // adjust field name as needed
    img = await loadImage(API_URL + imageUrl);

    await loadSimilarImages(imageUrl);
    
    return imageUrl;

  } catch (error) {
    console.error('Error loading image:', error);
  }
}

async function loadSimilarImages(url) {
    const filename = url.split('/').pop();
    try {
      const data = await postData(`${API_URL}/query/similar`, {
        url: filename,
        n_results: 16
      });
      urls = data.urls;
      const promises = data.urls.map(url => loadImage(API_URL + url));
      images = await Promise.all(promises);
      
    } catch (error) {
      console.error('Error loading similar images:', error);
    }
  }

function mousePressed() {
    // Check if click is in grid area
    if (mouseX > windowWidth - SIDEBAR_WIDTH) {
      const cols = Math.floor(SIDEBAR_WIDTH / (THUMBNAIL_SIZE + GRID_GAP));
      const col = Math.floor((mouseX - (windowWidth - SIDEBAR_WIDTH)) / (THUMBNAIL_SIZE + GRID_GAP));
      const row = Math.floor(mouseY / (THUMBNAIL_SIZE + GRID_GAP));
      const index = row * cols + col;
      
      if (index < images.length) {
        selectedImg = images[index];
      }
    }
}

async function postData(url, data) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      });
      return await response.json();
    } catch (error) {
      console.error('Error in POST request:', error);
      throw error;
    }
  }
