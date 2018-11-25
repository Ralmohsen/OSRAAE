# Copyright 2018 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


class ProgressBar:
    def __init__(self, total_iterations, prefix='Progress:', suffix='', decimals=1, length=70, fill='#'):
        self.format_string = "\r%s |%%s| %%.%df%%%% [%%d/%d] %s" % (prefix, decimals, total_iterations, suffix)
        self.total_iterations = total_iterations
        self.length = length
        self.fill = fill
        self.current_iteration = 0

    def increment(self, val=1):
        self.current_iteration += val
        percent = 100 * (self.current_iteration / float(self.total_iterations))
        filled_length = int(self.length * self.current_iteration // self.total_iterations)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(self.format_string % (bar, percent, self.current_iteration, ), end='\r')
        if self.current_iteration == self.total_iterations:
            print()

if __name__ == '__main__':
    from time import sleep

    items = list(range(0, 57))
    l = len(items)

    pb = ProgressBar(l)

    for i, item in enumerate(items):
        sleep(0.1)
        pb.increment()
