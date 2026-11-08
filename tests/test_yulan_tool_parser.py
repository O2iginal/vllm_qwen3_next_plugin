"""Regression tests against the installed vLLM parser and real YuLan tokenizer.
Run: .venv/bin/python -m unittest discover -s tests -p test_yulan_tool_parser.py -v
"""
import json
import unittest
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.yulan_tool_parser import YulanToolParser

MODEL='/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-Base-512k'
XML='<tool_call>\n<function=write>\n<parameter=path>\n/tmp/test.txt\n</parameter>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>'
HERMES='<tool_call>{"name":"write","arguments":{"path":"/tmp/test.txt","content":"hello"}}</tool_call>'
EXPECTED={'path':'/tmp/test.txt','content':'hello'}

class ParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
        cls.request=ChatCompletionRequest(model='test',messages=[{'role':'user','content':'test'}],tools=[{'type':'function','function':{'name':'write','parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}}}}}])

    def stream(self, chunks, parser=None):
        parser=parser or YulanToolParser(self.tokenizer)
        text=''; ids=[]; content=''; calls={}
        for chunk in [*chunks,'']:
            new_ids=self.tokenizer.encode(chunk,add_special_tokens=False) if chunk else [2]
            delta=parser.extract_tool_calls_streaming(text,text+chunk,chunk,ids,ids+new_ids,new_ids,self.request)
            text+=chunk;ids+=new_ids
            if delta:
                content+=delta.content or ''
                for call in delta.tool_calls or []:
                    entry=calls.setdefault(call.index,{'name':'','arguments':''})
                    if call.function:
                        entry['name']+=call.function.name or ''
                        entry['arguments']+=call.function.arguments or ''
        return content,calls,parser

    def assert_tools(self,chunks,count=1,content=''):
        prose,calls,p=self.stream(chunks)
        self.assertEqual(prose,content)
        self.assertEqual(len(calls),count)
        for i in range(count):
            self.assertEqual(calls[i]['name'],'write')
            self.assertEqual(json.loads(calls[i]['arguments']),EXPECTED)
        self.assertEqual(len(p.prev_tool_call_arr),count)
        return p

    def test_qwen_reported_chunks(self):
        self.assert_tools(['<tool_call>','\n','<','function','=','write','>','\n<parameter=path>\n/tmp/test.txt\n</parameter>','\n<parameter=content>\nhello\n</parameter>','\n</function>','\n</tool_call>'])
    def test_hermes_reported_chunks(self):
        self.assert_tools(['<tool_call>','{','"name":"write",','"arguments":{"path":"/tmp/test.txt","content":"hello"}}','</tool_call>'])
    def test_speculative_complete_xml(self): self.assert_tools([XML])
    def test_speculative_complete_hermes(self): self.assert_tools([HERMES])
    def test_xml_character_chunks(self): self.assert_tools(list(XML))
    def test_hermes_character_chunks(self): self.assert_tools(list(HERMES))
    def test_prose(self):
        for text in ['hello world','hello < world','literal <tool_cal']:
            content,calls,_=self.stream(list(text));self.assertEqual(content,text);self.assertFalse(calls)
    def test_prefix_xml(self): self.assert_tools(['Before. '+XML],content='Before. ')
    def test_prefix_hermes(self): self.assert_tools(['Before. '+HERMES],content='Before. ')
    def test_multiple_xml(self): self.assert_tools([XML+XML],count=2)
    def test_multiple_hermes(self): self.assert_tools([HERMES+HERMES],count=2)
    def test_missing_function_close(self): self.assert_tools([XML.replace('</function>','')])
    def test_nonstream(self):
        for text in (XML,HERMES,XML.replace('</function>','')):
            result=YulanToolParser(self.tokenizer).extract_tool_calls(text,self.request)
            self.assertTrue(result.tools_called);self.assertEqual(result.tool_calls[0].function.name,'write');self.assertEqual(json.loads(result.tool_calls[0].function.arguments),EXPECTED)
    def test_all_two_chunk_boundaries(self):
        for text in (XML, HERMES):
            for cut in range(1, len(text)):
                with self.subTest(format=text[:30], cut=cut):
                    self.assert_tools([text[:cut],text[cut:]])

    def test_fixed_size_speculative_chunks(self):
        for text in (XML+XML, HERMES+HERMES):
            for size in (2,3,7,13,31,64):
                with self.subTest(format=text[:30], size=size):
                    self.assert_tools([text[i:i+size] for i in range(0,len(text),size)],count=2)

    def test_suffix_prose(self):
        self.assert_tools([XML+' Done. <'],content=' Done. <')
        self.assert_tools([HERMES+' Done. <'],content=' Done. <')

    def test_nonstream_plain_text(self):
        result=YulanToolParser(self.tokenizer).extract_tool_calls('hello world',self.request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content,'hello world')

    def test_reuse(self):
        p=self.assert_tools([XML]);content,calls,_=self.stream([HERMES],p)
        self.assertEqual(content,'');self.assertEqual(len(calls),1);self.assertEqual(json.loads(calls[0]['arguments']),EXPECTED)

if __name__=='__main__': unittest.main()
