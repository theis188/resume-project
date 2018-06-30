from analyze_resumes import process_all
from config import specific_recommendations
import re

def create_list_of_justifications():
	ret = []
	for rec in specific_recommendations:
		new_rec = create_justification(rec)
		print(new_rec)
		ret.append( new_rec )
	return ret

def select_corpus(corpus,attrname,attrval):
	ind_list = [ k for k,obj in enumerate(attrs) if obj[attrname]==attrval ]
	return [corpus[k] for k in ind_list]

def create_justification(rec):
	from_term = {
				'nov_freq':find_frequency(rec['from'],nov_corp),
				'exp_freq':find_frequency(rec['from'],exp_corp)
				}
	to_term = {
				'nov_freq':find_frequency(rec['to'],nov_corp),
				'exp_freq':find_frequency(rec['to'],exp_corp)
				}
	justification_str = get_justification_str(rec['from'],rec['to'],from_term,to_term)
	rec['explanation'] = {
		'from_term':from_term,
		'to_term':to_term,
		'explanation_str':justification_str,
	}
	return rec	

def get_justification_str(from_str,to_str,from_term,to_term):
	ret = ('"'+
		to_str+'" appears {:.1f} times more frequently '+
		'in experienced resumes than novice'+
		' compared to '+from_str+'.'
	).format(
		(to_term['exp_freq']/to_term['nov_freq']) / (
		from_term['exp_freq']/from_term['nov_freq'])
	)
	return ret

def find_frequency(s,subcorpus):
	total = len(subcorpus)
	present = sum( [ len( re.findall(s,ci) ) for ci in subcorpus ])
	return 1.*present/total

def write_json_list_of_expectations():
	to_write = create_list_of_justifications()
	with open('from_to_justifications.json','w') as f:
		f.write(json.dumps(to_write))

attrs,corpus,sentences = process_all()
levels = ['Less than 1 year', '3-5 years' ,'More than 10 years']
nov_corp = select_corpus(corpus,'level',levels[0])
exp_corp = select_corpus(corpus,'level',levels[2])
l = create_list_of_justifications()
