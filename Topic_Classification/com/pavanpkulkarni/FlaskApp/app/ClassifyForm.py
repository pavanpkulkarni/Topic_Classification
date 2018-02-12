from flask_wtf import Form
from wtforms import SubmitField, StringField

class ClassificationTitleForms(Form):
  key = StringField("Title")
  submit = SubmitField("Submit")
  topic_taxonomy = SubmitField("Topic Taxonomy")
