from datetime import datetime

from flask import Blueprint, url_for, request
from werkzeug.utils import redirect

from DBWEB.models.models import Question, Answer
from DBWEB import DB

answerBP = Blueprint('answer',
                     __name__,
                     url_prefix='/answer')

@answerBP.route('/create/<int:question_id>', methods=('POST',))

def create(question_id):
    question = Question.query.get_or_404(question_id)
    content = request.form['content']
    answer = Answer(content=content, create_date=datetime.now())
    question.answer_set.append(answer)
    DB.session.commit()
    return redirect(url_for('question.detail', question_id=question_id))