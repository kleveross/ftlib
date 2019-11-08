# By default we use 3.6+ version. If you have a special case
# where you MUST use python 2, make sure you explain clearly
# the reasons before doing so.
FROM python:3.6

# add other labels as well
LABEL maintainer="Jiayu Liu <jiayu@caicloud.io>"

# make sure your unicode works!
ENV LC_ALL="C.UTF-8" \
  LANG="C.UTF-8"

WORKDIR /opt/app/

# copy and install first to enable layer caching
COPY Pipfile Pipfile.lock /opt/app/

RUN pip3 install pipenv && \
  pipenv install --deploy --system

# copy the rest of code
# if you have large data you do not want to include, create a
# .dockerignore file in root.
COPY . /opt/app/

ENV PYTHONPATH=/opt/app/:${PYTHONPATH}

# make sure it matches below
EXPOSE 8080

# note that this docker file assumes that you are building a flask
# application which is an excellent python web framework.
# also since flask's built-in web server is for debug only
# we will use more general WSGI container e.g. twisted to
# enable better throughput.
#
# see http://flask.pocoo.org/docs/0.12/deploying/#deployment
CMD ["twistd", "-n", "web", "--port", "tcp:port=8080", "--wsgi", "web.app"]
