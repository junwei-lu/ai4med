# Contributing

We welcome any contributions to improve the quality of the course. If you discover any typos, broken links, missing content, textual ambiguities, unclear explanations, or unreasonable text structures, please assist us in making corrections to provide readers with better quality learning resources.


### Content fine-tuning

You can follow these steps to modify text or code.

1. Open the our course [github repository](https://github.com/junwei-lu/bst236). Find the Markdown source file you want to modify.
2. Modify the Markdown source file content, check the accuracy of the content, and try to keep the formatting consistent.
3. Fill in the modification description at the bottom of the page, then click the "Propose file change" button. After the page redirects, click the "Create pull request" button to initiate the pull request.

Figures cannot be directly modified and require the creation of a new [Issue](https://github.com/krahets/hello-algo/issues) or a comment to describe the problem. We will redraw and replace the figures as soon as possible.

### Content creation

If you are interested in participating in this open-source project, including translating code into other programming languages or expanding article content, then the following Pull Request workflow needs to be implemented.

1. Log in to GitHub and Fork the [code repository](https://github.com/junwei-lu/bst236) of this book to your personal account.
2. Go to your Forked repository web page and use the `git clone` command to clone the repository to your local machine.
3. Create content locally and perform complete tests to verify the correctness of the code.
4. Commit the changes made locally, then push them to the remote repository.
5. Refresh the repository webpage and click the "Create pull request" button to initiate the pull request.

### Docker deployment

In the `bst236` root directory, execute the following Docker script to access the project at `http://localhost:8000`:

```shell
docker-compose up -d
```

Use the following command to remove the deployment:

```shell
docker-compose down
```

# Resources

- This course website is built on the infrastructure of a fascinating open-source project [Hello Algo](www.hello-algo.com/en/). Many materials of this course are adapted from their book. We thank the authors for their selfless contributions to the open-source community.
- This course website is build on the package of [MkDocs](https://www.mkdocs.org/). MkDocs is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. You can learn more about it [here](https://www.mkdocs.org/user-guide/).