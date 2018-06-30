from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
import string
import time
import random
import json
import os
import argparse
import newspaper
import re
import random
import math

wait_time = 5

records_root = '/Users/matthewtheisen/Google_Drive/Careers/Data/clean/repo/records/'
json_files = os.listdir(records_root)
json_files.remove('.DS_Store')
done_records = os.listdir('/Users/matthewtheisen/Google_Drive/Careers/Data/clean/repo/resumes/records/')
done_titles = set([t.split('.')[0] for t in done_records])

titles = []
for json_file in json_files:
	with open(records_root+json_file) as f:
		jline = json.loads(next(f))
		titles.append(jline['search_term'])

address = 'https://www.indeed.com/resumes/'
levels = ['3-5 years', 'Less than 1 year', 'More than 10 years']
one_title = titles[0]

def get_random(t_min=2,t_max=10,fun=lambda x: math.exp(math.exp(x)) ):
	im_min=fun(0)
	im_max=fun(1)
	ret = fun(random.random())
	ret = 1.*(ret-im_min)*(t_max-t_min)/(im_max-im_min)+t_min
	return ret

def get_num_waits(n_min=400,n_max=400):
	return int( n_min+( random.random() *( n_max-n_min ) ) )

def main():
	for title in titles:
		if title.replace('/','') in done_titles:
			continue
		f = open('records/' + title.replace('/','') + '.json', 'w')
		for level in levels:
			get_level(f,title,level)
		f.close()

def get_level(f,title,level):
	n_tries = 0
	while n_tries<5:
		ff = webdriver.Firefox()
		ff.get(address)
		time.sleep(wait_time)
		box = ff.find_element_by_id('query') 
		ff.find_element_by_id('location').clear()
		keyword = title.lower().replace(',','').replace('/',' ')
		box.clear()
		box.send_keys(keyword)
		box.send_keys(Keys.RETURN)
		time.sleep(wait_time)
		try:
			ff.find_elements_by_css_selector('input[data-tn-element^="'+level+'"]')[0].click()
			save(title,ff,level,f)
			ff.quit()
			return
		except IndexError:
			ff.quit()
			print('throttled. sleep.')
			tot = 0
			for i in range( get_num_waits() ):
				print( int( tot ) )
				sleep_time = 10
				time.sleep(sleep_time)
				n_tries += 1
				tot = sleep_time+tot
				# time.sleep(600)

def save(title,ff,level,f):
	try:
		x_button = ff.find_element_by_id('popover-x-button')
		x_button.click()
	except:
		pass
	time.sleep(get_random())
	title_links = ff.find_elements_by_css_selector('.app_link')
	for link in title_links:
		res_title = link.text
		print(res_title)
		link.click()
		time.sleep(get_random())
		ff.switch_to.window ( ff.window_handles[1] )
		if 'indeed.com' in ff.current_url:
			# bulk_text = [el.text for el in ff.find_elements_by_css_selector('#job_summary p') if len(el.text) > 200]
			# bullets = [el.text for el in ff.find_elements_by_css_selector('#job_summary ul li') ]
			obj = {}
			obj['title'] = res_title
			obj['summary'] = selector_filter(ff,'#res_summary', lambda x: x.text )
			obj['work_experience'] = selector_filter(ff,'.section-item.workExperience-content',get_experience)
			obj['education'] = selector_filter(ff,'.section-item.education-content',get_education)
			obj['level']=level
			f.write( json.dumps(obj) + '\n' )
		time.sleep(get_random())
		ff.switch_to.window( ff.window_handles[0] )
	time.sleep(get_random())

def selector_filter(ff, selector, fun):
	ff_obj = ff.find_elements_by_css_selector(selector)
	if len(ff_obj)==0:
		return ''
	else:
		return fun(ff_obj[0])

def get_first_css(selector,ff_obj):
	retlist=ff_obj.find_elements_by_css_selector(selector)
	if len(retlist)==0:
		return ''
	else:
		return retlist[0].text

def get_experience(ff_obj):
	ret = []
	for record in ff_obj.find_elements_by_css_selector('.data_display'):
		ret.append({
			'title':get_first_css('.work_title',record),
			'company':get_first_css('.work_company',record),
			'description':get_first_css('.work_description',record)
		})
	return ret

def get_education(ff_obj):
	ret = []
	for record in ff_obj.find_elements_by_css_selector('.data_display'):
		ret.append({
			'title':get_first_css('.edu_title',record),
			'school':get_first_css('.edu_school',record)
		})
	return ret



if __name__ == '__main__':
	# save(one_title)
	main()
	# ff.stop()
	