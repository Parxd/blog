---
title: "a learning journey through ML systems"
date: 2026-01-25T17:12:11+07:00
comment: true
tags: ["personal"]
---

As I approach around a year since formally starting my journey of self-learning and stumbling through the somewhat vague world of ML systems, I wanted to look back and recap / document my entire roadmap and the insane rabbit holes I've fallen through during the past year, and also outline what my goals are for the next couple months. 

The real story started in my sophomore year of college, when I started my project GroundUpNeuralNet, a *very* minimal implementation of an MLP library I wrote in C++ after taking a DSA class. At this point, I wasn't very concerned (or knowledgeable) about performance, so I leaned heavily on Eigen for the linear algebra. Nevertheless, I still learned a good deal about multivariate calc, backprop, and just generally how to structure a relatively large C++ project.
You can still play around with it today [here](https://github.com/Parxd/GroundUpNeuralNet), although it only supports linear layers and a handful of activation + loss functions.

During my exploration of how to improve this project, I learned about the concept of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), or autodiff, which greatly opened my eyes as to how actual production ML libraries implement backprop. And of course, I wanted to write a toy autodiff system myself, which led to another project: [accelerate](https://github.com/Parxd/accelerate). Using NumPy + CuPy for the backend linear algebra, I wrote a simple autodiff library centered around a toy tensor implementation. Being based on autodiff instead of individually written complex gradients, this allowed for chaining of more complex ops. However, I was far too focused on designing a interface inspired PyTorch: with the main `Tensor` abstraction, naturally calling overloaded operators that build the autodiff DAG, and ultimately calling `backward`  on some leaf tensor. 

Disappointingly, when I actually benchmarked my "library's" performance against PyTorch, I found it to be many magnitudes slower on the GPU, even though while using CuPy for the compute. Upon profiling training runs, I found many issues: poor core design choices (i.e. *very* tightly coupled interface and backend, Python-level DAG traversal during backprop) caused frequent device syncing, slow and unnecessary device memory movement, and individual kernel launch overhead to dominate. I tried to "hack" CuPy to fix these issues, but I ultimately found that my design choices would make it really difficult. I thus turned myself to a lower level on the stack: CUDA / C++.

```python {title="hello", style=onedark}
def backward(self, grad: Optional[Tensor] = None) -> None:
    if self.requires_grad:
        if grad is None:
            if self.ndim != 0:
                raise RuntimeError("grad implicitly created only for scalar outputs")
            else:
                grad = Tensor(1, device='cpu' if self.device is CPU else 'cuda')
        self.grad.data += grad.data
        if self.grad_fn:
            gradients = self.grad_fn(self.grad.data)
            for child, gradient in zip(self._children, gradients):
                # re-wrap incoming gradient array as Tensor
                child.backward(Tensor(gradient, device='cpu' if self.device is CPU else 'cuda'))
```

During the winter break of my final year in college, I unfortunately had no other choice but to stay on campus, but I took this time as an opportunity to finally begin learning CUDA kernel programming.
