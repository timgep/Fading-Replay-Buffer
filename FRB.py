class FadingReplayBuffer:
    def __init__(self, capacity=1280000):
        self.capacity = capacity
        self.cache, self.indices = [], []
        self.buffer, self.length = deque(maxlen=capacity), 0
        self.x, self.step, self.s = 0.0, 1/self.capacity, 1.0
        

    # priority for old memories are fading gradually
    def fade(self, norm_index):
        return (1.001-self.s)*np.tanh(14*norm_index**2)

    #adds average between two to store more data (except transitions with dones)
    def add_average(self, transition):
        self.cache.append(transition)

        if len(self.cache)>=2:
            transition = self.cache[0]
            if self.cache[0][-1] == True or self.cache[1][-1] == True:
                del self.cache[0]
            else:
                for j, (x,y) in enumerate(zip(self.cache[0], self.cache[1])):
                    transition[j] = (x+y)/2
                self.cache = []
            self.add(transition)

    # adds to buffer
    def add(self, transition):
        self.buffer.append(transition)
        self.length = len(self.buffer)
        if self.length < self.capacity:
            self.indices.append(self.length-1)

        self.x += self.step
        self.s = math.exp(-self.x)

    # samples big batch then re-samples smaller batch with less priority to old data
    def sample(self, batch_size, device, CER=False):
        if len(self.buffer) >= 1024:
            
            #batch = random.sample(self.buffer, k=batch_size)

            sample_indices = random.sample(self.indices, k=1024)
            probs = self.fade(np.array(sample_indices)/self.length)
            batch_indices = np.random.default_rng().choice(sample_indices, p=probs/np.sum(probs), size=batch_size, replace=False)
            batch = [self.buffer[indx-1] for indx in batch_indices]

            if CER: batch.append(self.buffer[-1])
            states, actions, rewards, next_states, dones = map(np.vstack, zip(*batch))
            
            return (
                torch.FloatTensor(states).to(device),
                torch.FloatTensor(actions).to(device),
                torch.FloatTensor(rewards).to(device),
                torch.FloatTensor(next_states).to(device),
                torch.FloatTensor(dones).to(device),
            )

            

    def __len__(self):
        return self.length
